#include "image.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace npuforge {
namespace {
constexpr U64 Limit = 64ULL * 1024 * 1024;
const std::array<std::string, 10> Names{"",          "ALLOC",     "FREE",
                                        "DMA_LOAD",  "DMA_STORE", "MATMUL",
                                        "ADD",       "RELU",      "FUSED_MATMUL_ADD_RELU",
                                        "PIM_MATMUL"};
U64 plus(U64 a, U64 b) {
  if (b > std::numeric_limits<U64>::max() - a)
    throw std::invalid_argument("address overflow");
  return a + b;
}
U64 times(U64 a, U64 b) {
  if (a && b > std::numeric_limits<U64>::max() / a)
    throw std::invalid_argument("size overflow");
  return a * b;
}
U64 integer(const std::string &s, U64 at, unsigned width) {
  if (plus(at, width) > s.size())
    throw std::invalid_argument("truncated instruction image");
  U64 value = 0;
  for (unsigned j = 0; j < width; ++j)
    value |= U64(static_cast<unsigned char>(s[at + j])) << (8 * j);
  return value;
}
float real(const std::string &s, U64 at) {
  std::uint32_t bits = static_cast<std::uint32_t>(integer(s, at, 4));
  float f;
  std::memcpy(&f, &bits, 4);
  if (!std::isfinite(f))
    throw std::invalid_argument("non-finite FP32 value");
  return f;
}
void append_real(std::string &s, float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, 4);
  for (unsigned j = 0; j < 4; ++j)
    s.push_back(static_cast<char>((bits >> (8 * j)) & 255));
}
U64 elements(const TensorImage &t) {
  U64 n = 1;
  if (t.shape.size() > 64)
    throw std::invalid_argument("tensor rank exceeds 64");
  for (U64 dim : t.shape) {
    if (!dim)
      throw std::invalid_argument("zero tensor dimension");
    n = times(n, dim);
  }
  return n;
}
struct Word {
  int owner = -1;
  U64 ready = 0, reader_end = 0;
  unsigned source = NoTensor;
  U64 index = 0, reduction_end = 0, reduction_target = 0;
};
struct Memory {
  std::vector<float> values;
  std::vector<Word> state;
  explicit Memory(U64 bytes) : values(bytes / 4), state(bytes / 4) {}
};
} // namespace

std::vector<ImageInstruction> decode_image(const std::string &data) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  if (data.size() < 32 || data.compare(0, 8, std::string("NPUFIMG\0", 8)) != 0)
    throw std::invalid_argument("invalid instruction image magic/header");
  if (integer(data, 8, 2) != 1 || integer(data, 10, 2) != 256 || integer(data, 12, 4) != 1)
    throw std::invalid_argument("unsupported binary version, record length or precision");
  U64 n = integer(data, 16, 8), deps = integer(data, 24, 8);
  U64 pool = plus(32, times(n, 256));
  if (n > 1000000 || data.size() > 256 * 1024 * 1024ULL ||
      plus(pool, times(deps, 4)) != data.size())
    throw std::invalid_argument("invalid binary image length/counts");
  std::vector<ImageInstruction> result;
  result.reserve(static_cast<std::size_t>(n));
  U64 expected_dep = 0;
  for (U64 j = 0; j < n; ++j) {
    U64 at = 32 + 256 * j;
    unsigned op = static_cast<unsigned>(integer(data, at, 4));
    if (!op || op >= Names.size() || integer(data, at + 4, 4) != j)
      throw std::invalid_argument("invalid opcode or instruction identity");
    ImageInstruction i;
    i.timing.id = static_cast<int>(j);
    i.timing.op = Names[op];
    i.flags = static_cast<unsigned>(integer(data, at + 8, 4));
    i.mode = static_cast<unsigned>(integer(data, at + 12, 4));
    if (i.flags & ~63U || i.mode > 5 || integer(data, at + 28, 4) != 1)
      throw std::invalid_argument("invalid flags, addressing mode or dtype");
    U64 buffer = integer(data, at + 16, 4);
    if (buffer > static_cast<U64>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("compiled instructions require a nonnegative buffer ID");
    i.timing.buffer = static_cast<int>(buffer);
    U64 begin = integer(data, at + 20, 4), count = integer(data, at + 24, 4);
    if (begin != expected_dep || plus(begin, count) > deps)
      throw std::invalid_argument("invalid dependency pool range");
    expected_dep += count;
    std::unordered_set<int> seen;
    for (U64 d = 0; d < count; ++d) {
      U64 id = integer(data, pool + (begin + d) * 4, 4);
      if (id >= n || id == j || !seen.insert(static_cast<int>(id)).second)
        throw std::invalid_argument("invalid binary dependency");
      i.timing.deps.push_back(static_cast<int>(id));
    }
    for (unsigned k = 0; k < 4; ++k) {
      i.tensors[k] = static_cast<unsigned>(integer(data, at + 32 + k * 4, 4));
      i.addresses[k] = integer(data, at + 48 + k * 8, 8);
    }
    i.timing.sram_bytes = integer(data, at + 80, 8);
    i.timing.working_set_bytes = integer(data, at + 88, 8);
    i.timing.bytes = integer(data, at + 96, 8);
    i.timing.macs = integer(data, at + 104, 8);
    i.timing.elements = integer(data, at + 112, 8);
    i.timing.local_read_bytes = integer(data, at + 120, 8);
    i.flat_start = integer(data, at + 128, 8);
    i.count = integer(data, at + 136, 8);
    for (unsigned k = 0; k < 6; ++k)
      i.dims[k] = static_cast<unsigned>(integer(data, at + 144 + k * 4, 4));
    i.alpha = real(data, at + 168);
    i.beta = real(data, at + 172);
    for (U64 k = at + 176; k < at + 256; ++k)
      if (data[k])
        throw std::invalid_argument("nonzero reserved instruction bytes");
    result.push_back(std::move(i));
  }
  if (expected_dep != deps)
    throw std::invalid_argument("unused dependency bytes");
  std::vector<std::vector<unsigned>> children(result.size());
  std::vector<unsigned> counts(result.size());
  std::queue<unsigned> ready;
  for (unsigned j = 0; j < result.size(); ++j) {
    counts[j] = static_cast<unsigned>(result[j].timing.deps.size());
    if (!counts[j])
      ready.push(j);
    for (int d : result[j].timing.deps)
      children[d].push_back(j);
    for (U64 address : result[j].addresses)
      if (address % 4)
        throw std::invalid_argument("unaligned binary address");
  }
  std::size_t visited = 0;
  while (!ready.empty()) {
    auto id = ready.front();
    ready.pop();
    ++visited;
    for (auto child : children[id])
      if (--counts[child] == 0)
        ready.push(child);
  }
  if (visited != result.size())
    throw std::invalid_argument("cyclic binary dependencies");
  return result;
}

ImageResult execute_image(const std::string &bytes, const std::string &weights,
                          const std::vector<TensorImage> &tensors,
                          const std::map<unsigned, std::string> &inputs, U64 dram_bytes,
                          const Hardware &hardware) {
  if (dram_bytes > Limit || dram_bytes % 4 || hardware.sram_bytes > Limit ||
      hardware.sram_bytes < 4 || weights.size() > dram_bytes || weights.size() % 4)
    throw std::invalid_argument("invalid image memory capacity (maximum 64 MiB per space)");
  auto instructions = decode_image(bytes);
  for (std::size_t a = 0; a < tensors.size(); ++a) {
    const auto &t = tensors[a];
    if (t.address % 64 || times(elements(t), 4) != t.size || !t.size ||
        t.allocation_size != (plus(t.size, 63) / 64) * 64 || t.first > t.last ||
        plus(t.address, t.allocation_size) > dram_bytes || t.role_bits > 7 ||
        ((t.role_bits & 3) == 3))
      throw std::invalid_argument("invalid tensor allocation/shape");
    for (std::size_t b = 0; b < a; ++b) {
      const auto &other = tensors[b];
      if (t.address < plus(other.address, other.allocation_size) &&
          other.address < plus(t.address, t.allocation_size) && t.first <= other.last &&
          other.first <= t.last)
        throw std::invalid_argument("overlapping live DRAM allocations");
    }
  }
  Memory dram(dram_bytes), sram((hardware.sram_bytes / 4) * 4);
  unsigned required_inputs = 0;
  for (unsigned id = 0; id < tensors.size(); ++id) {
    const auto &t = tensors[id];
    const std::string *data = nullptr;
    if (t.role_bits & 1) {
      if (plus(t.address, t.size) > weights.size())
        throw std::invalid_argument("weight image is too short");
      for (U64 j = 0; j < t.size / 4; ++j) {
        U64 pos = t.address / 4 + j;
        dram.values[pos] = real(weights, t.address + j * 4);
        dram.state[pos].owner = static_cast<int>(id);
      }
    }
    if (t.role_bits & 2) {
      ++required_inputs;
      auto it = inputs.find(id);
      if (it == inputs.end() || it->second.size() != t.size)
        throw std::invalid_argument("missing or incorrectly sized runtime input");
      data = &it->second;
      for (U64 j = 0; j < t.size / 4; ++j) {
        U64 pos = t.address / 4 + j;
        dram.values[pos] = real(*data, j * 4);
        dram.state[pos].owner = static_cast<int>(id);
      }
    }
  }
  if (required_inputs != inputs.size())
    throw std::invalid_argument("unexpected runtime input");
  std::vector<Instruction> timing;
  for (const auto &i : instructions)
    timing.push_back(i.timing);
  ImageResult result;
  result.timing = simulate(timing, hardware);
  std::map<int, std::pair<U64, U64>> allocations;
  auto tensor = [&](unsigned id) -> const TensorImage & {
    if (id >= tensors.size())
      throw std::invalid_argument("invalid tensor reference");
    return tensors[id];
  };
  for (const auto &event : result.timing.trace) {
    const auto &i = instructions[event.id];
    const auto &t = i.timing;
    auto op = t.op;
    if (op == "ALLOC") {
      U64 begin = i.addresses[3], end = plus(begin, t.sram_bytes);
      if (begin % 4 || end > hardware.sram_bytes || t.sram_bytes % 4)
        throw std::invalid_argument("SRAM allocation out of bounds/alignment");
      for (const auto &[id, range] : allocations) {
        (void)id;
        if (begin < range.second && range.first < end)
          throw std::invalid_argument("overlapping live SRAM allocations");
      }
      allocations[t.buffer] = {begin, end};
      for (U64 p = begin / 4; p < end / 4; ++p)
        sram.state[p] = {};
      continue;
    }
    if (op == "FREE") {
      allocations.erase(t.buffer);
      continue;
    }
    auto arena = allocations.find(t.buffer);
    if (arena == allocations.end())
      throw std::invalid_argument("missing active SRAM buffer");
    std::unordered_set<Word *> reads;
    auto sram_pos = [&](U64 address) -> U64 {
      if (address % 4 || address < arena->second.first || plus(address, 4) > arena->second.second)
        throw std::invalid_argument("SRAM operand out of allocated range");
      return address / 4;
    };
    auto read = [&](Memory &mem, U64 pos, int owner) -> float {
      if (pos >= mem.values.size() || mem.state[pos].owner != owner)
        throw std::invalid_argument("read before initialization or stale aliased tensor");
      auto &state = mem.state[pos];
      if (state.ready > event.start)
        throw std::invalid_argument("read before transfer/compute completion");
      reads.insert(&state);
      return mem.values[pos];
    };
    auto write = [&](Memory &mem, U64 pos, int owner, float value) {
      if (pos >= mem.values.size() || !std::isfinite(value))
        throw std::invalid_argument("invalid/non-finite memory write");
      auto &state = mem.state[pos];
      if (state.ready > event.start || state.reader_end > event.start)
        throw std::invalid_argument("memory overwrite while instruction is outstanding");
      mem.values[pos] = value;
      state.owner = owner;
      state.ready = event.end;
    };
    auto rd = [&](unsigned tid, U64 index) {
      const auto &desc = tensor(tid);
      if (index >= desc.size / 4)
        throw std::invalid_argument("tensor read out of bounds");
      return read(dram, plus(desc.address / 4, index), static_cast<int>(tid));
    };
    auto rs = [&](U64 base, U64 index, unsigned tid, U64 logical) {
      U64 pos = sram_pos(plus(base, times(index, 4)));
      if (sram.state[pos].source != tid || sram.state[pos].index != logical)
        throw std::invalid_argument("staged operand provenance mismatch");
      return read(sram, pos, t.buffer);
    };
    auto ws = [&](U64 base, U64 index, float value, unsigned tid, U64 logical, U64 reduction = 0,
                  U64 target = 0) {
      U64 pos = sram_pos(plus(base, times(index, 4)));
      write(sram, pos, t.buffer, value);
      sram.state[pos].source = tid;
      sram.state[pos].index = logical;
      sram.state[pos].reduction_end = reduction;
      sram.state[pos].reduction_target = target;
    };
    auto wd = [&](unsigned tid, U64 index, float value) {
      const auto &desc = tensor(tid);
      if (index >= desc.size / 4)
        throw std::invalid_argument("tensor write out of bounds");
      write(dram, plus(desc.address / 4, index), static_cast<int>(tid), value);
    };
    U64 m0 = i.dims[0], n0 = i.dims[1], k0 = i.dims[2], m = i.dims[3], n = i.dims[4], k = i.dims[5];
    auto broadcast_index = [&](unsigned source, U64 flat) {
      const auto &shape = tensor(source).shape;
      const auto &out = tensor(i.tensors[3]).shape;
      if (shape.size() > out.size() || flat >= elements(tensor(i.tensors[3])))
        throw std::invalid_argument("invalid broadcast dimensions/range");
      U64 index = 0, stride = 1;
      for (std::size_t axis = out.size(); axis > 0; --axis) {
        U64 coord = flat % out[axis - 1];
        flat /= out[axis - 1];
        U64 dim =
            axis + shape.size() > out.size() ? shape[axis + shape.size() - out.size() - 1] : 1;
        if (dim != 1 && dim != out[axis - 1])
          throw std::invalid_argument("incompatible broadcast shape");
        if (dim != 1)
          index = plus(index, times(coord, stride));
        stride = times(stride, dim);
      }
      return index;
    };
    auto matrix_index = [&](unsigned tid, bool transpose, U64 row, U64 col) {
      const auto &shape = tensor(tid).shape;
      if (shape.size() != 2)
        throw std::invalid_argument("matrix operand must have rank 2");
      if (transpose)
        std::swap(row, col);
      if (row >= shape[0] || col >= shape[1])
        throw std::invalid_argument("matrix tile out of bounds");
      return plus(times(row, shape[1]), col);
    };
    auto disjoint_output = [&](U64 start, U64 length) {
      U64 out_end = plus(i.addresses[3], times(times(m, n), 4));
      if (start < out_end && i.addresses[3] < plus(start, length))
        throw std::invalid_argument("compute output aliases a staged input");
    };
    if (op == "DMA_LOAD") {
      if (i.addresses[0] != tensor(i.tensors[0]).address)
        throw std::invalid_argument("DMA source address does not match tensor allocation");
      U64 moved = 0;
      if (i.mode <= 1) {
        U64 rows = i.mode == 0 ? m : k, cols = i.mode == 0 ? k : n;
        if (!rows || !cols)
          throw std::invalid_argument("empty matrix transfer");
        for (U64 a = 0; a < rows; ++a)
          for (U64 b = 0; b < cols; ++b) {
            auto index = matrix_index(i.tensors[0], (i.flags & (i.mode == 0 ? 16U : 32U)) != 0,
                                      (i.mode == 0 ? m0 : k0) + a, (i.mode == 0 ? k0 : n0) + b);
            ws(i.addresses[3], a * cols + b, rd(i.tensors[0], index), i.tensors[0], index);
          }
        moved = times(times(rows, cols), 4);
      } else if (i.mode == 2 || i.mode == 3) {
        U64 count = i.mode == 2 ? times(m, n) : i.count;
        if (!count)
          throw std::invalid_argument("empty broadcast transfer");
        std::set<U64> sources;
        const auto &out = tensor(i.tensors[3]);
        if (i.mode == 2 && out.shape.size() != 2)
          throw std::invalid_argument("invalid bias output rank");
        for (U64 j = 0; j < count; ++j) {
          U64 flat = i.mode == 2 ? plus(times(m0 + j / n, out.shape[1]), n0 + j % n)
                                 : plus(i.flat_start, j);
          U64 source = broadcast_index(i.tensors[0], flat);
          sources.insert(source);
          ws(i.addresses[3], j, rd(i.tensors[0], source), i.tensors[0], source);
        }
        moved = times(sources.size(), 4);
      } else
        throw std::invalid_argument("invalid DMA load mode");
      if (t.bytes != moved)
        throw std::invalid_argument("DMA load byte count mismatch");
    } else if (op == "DMA_STORE") {
      const auto &out = tensor(i.tensors[3]);
      if (i.addresses[3] != out.address)
        throw std::invalid_argument("DMA destination address mismatch");
      U64 count = i.mode == 4 ? times(m, n) : i.count;
      if (!count || (i.mode != 4 && i.mode != 5) || t.bytes != times(count, 4))
        throw std::invalid_argument("invalid DMA store dimensions/byte count");
      if (i.mode == 4 && out.shape.size() != 2)
        throw std::invalid_argument("matrix store rank mismatch");
      for (U64 j = 0; j < count; ++j) {
        U64 flat = i.mode == 4 ? matrix_index(i.tensors[3], false, m0 + j / n, n0 + j % n)
                               : plus(i.flat_start, j);
        const auto &state = sram.state[sram_pos(plus(i.addresses[0], times(j, 4)))];
        if (state.reduction_end != state.reduction_target)
          throw std::invalid_argument("store before matrix reduction is complete");
        wd(i.tensors[3], flat, rs(i.addresses[0], j, i.tensors[3], flat));
      }
    } else if (op == "MATMUL" || op == "PIM_MATMUL" || op == "FUSED_MATMUL_ADD_RELU") {
      if (!m || !n || !k || t.macs != times(times(m, n), k))
        throw std::invalid_argument("invalid matrix work dimensions");
      disjoint_output(i.addresses[0], times(times(m, k), 4));
      bool pim = op == "PIM_MATMUL", final = (i.flags & 2) != 0;
      const auto &left_shape = tensor(i.tensors[0]).shape;
      const auto &right_shape = tensor(i.tensors[1]).shape;
      const auto &output_shape = tensor(i.tensors[3]).shape;
      if (left_shape.size() != 2 || right_shape.size() != 2 || output_shape.size() != 2)
        throw std::invalid_argument("invalid matrix tensor rank");
      U64 rows = left_shape[(i.flags & 16) ? 1 : 0];
      U64 reduction = left_shape[(i.flags & 16) ? 0 : 1];
      U64 columns = right_shape[(i.flags & 32) ? 0 : 1];
      if (right_shape[(i.flags & 32) ? 1 : 0] != reduction || output_shape[0] != rows ||
          output_shape[1] != columns || plus(m0, m) > rows || plus(n0, n) > columns ||
          plus(k0, k) > reduction || bool(i.flags & 1) != (k0 != 0) || ((i.flags & 4) && !final))
        throw std::invalid_argument("invalid matrix shape, reduction or activation order");
      bool needs_epilogue = (i.flags & 8) || i.tensors[2] != NoTensor || i.alpha != 1 || final;
      if (needs_epilogue && final != (k0 + k == reduction))
        throw std::invalid_argument("epilogue must execute after the final reduction tile");
      if (pim) {
        if (i.addresses[1] != tensor(i.tensors[1]).address ||
            t.local_read_bytes != times(times(k, n), 4))
          throw std::invalid_argument("invalid PIM operand address/local traffic");
      } else
        disjoint_output(i.addresses[1], times(times(k, n), 4));
      bool bias = i.tensors[2] != NoTensor;
      U64 scalar = (i.alpha != 1) + (bias ? 1 + (i.beta != 1) : 0) + ((i.flags & 4) != 0);
      if (t.elements != (final ? times(times(m, n), scalar) : 0))
        throw std::invalid_argument("matrix epilogue work mismatch");
      if (bias && final)
        disjoint_output(i.addresses[2], times(times(m, n), 4));
      for (U64 a = 0; a < m; ++a)
        for (U64 b = 0; b < n; ++b) {
          U64 index = a * n + b;
          U64 logical = matrix_index(i.tensors[3], false, m0 + a, n0 + b);
          if ((i.flags & 1) && sram.state[sram_pos(i.addresses[3] + index * 4)].reduction_end != k0)
            throw std::invalid_argument("noncontiguous accumulator reduction");
          float acc = i.flags & 1 ? rs(i.addresses[3], index, i.tensors[3], logical) : 0.0F;
          for (U64 c = 0; c < k; ++c) {
            float right = pim ? rd(i.tensors[1],
                                   matrix_index(i.tensors[1], (i.flags & 32) != 0, k0 + c, n0 + b))
                              : rs(i.addresses[1], c * n + b, i.tensors[1],
                                   matrix_index(i.tensors[1], (i.flags & 32) != 0, k0 + c, n0 + b));
            acc += rs(i.addresses[0], a * k + c, i.tensors[0],
                      matrix_index(i.tensors[0], (i.flags & 16) != 0, m0 + a, k0 + c)) *
                   right;
          }
          if (final) {
            acc *= i.alpha;
            if (bias)
              acc += i.beta * rs(i.addresses[2], index, i.tensors[2],
                                 broadcast_index(i.tensors[2], logical));
            if (i.flags & 4)
              acc = std::max(acc, 0.0F);
          }
          ws(i.addresses[3], index, acc, i.tensors[3], logical, k0 + k, reduction);
        }
    } else if (op == "ADD" || op == "RELU") {
      if (!i.count || t.elements != i.count)
        throw std::invalid_argument("invalid vector work count");
      // Vector output occupies its own staging interval.
      for (unsigned operand = 0; operand < (op == "ADD" ? 2U : 1U); ++operand)
        if (i.addresses[operand] < plus(i.addresses[3], times(i.count, 4)) &&
            i.addresses[3] < plus(i.addresses[operand], times(i.count, 4)))
          throw std::invalid_argument("vector output aliases input staging");
      for (U64 j = 0; j < i.count; ++j) {
        U64 logical = plus(i.flat_start, j);
        float value = rs(i.addresses[0], j, i.tensors[0], broadcast_index(i.tensors[0], logical));
        value = op == "ADD" ? value + rs(i.addresses[1], j, i.tensors[1],
                                         broadcast_index(i.tensors[1], logical))
                            : std::max(value, 0.0F);
        ws(i.addresses[3], j, value, i.tensors[3], logical);
      }
    } else
      throw std::invalid_argument("unsupported executable instruction");
    for (auto *state : reads)
      state->reader_end = std::max(state->reader_end, event.end);
  }
  for (unsigned tid = 0; tid < tensors.size(); ++tid)
    if (tensors[tid].role_bits & 4) {
      std::string output;
      const auto &desc = tensors[tid];
      output.reserve(static_cast<std::size_t>(desc.size));
      for (U64 j = 0; j < desc.size / 4; ++j) {
        U64 pos = desc.address / 4 + j;
        if (dram.state[pos].owner != static_cast<int>(tid))
          throw std::invalid_argument("output tensor incomplete or overwritten");
        append_real(output, dram.values[pos]);
      }
      result.outputs[tid] = std::move(output);
    }
  return result;
}
} // namespace npuforge
