#include "image.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace npuforge;
namespace {
Hardware hardware(const py::dict &config) {
  Hardware h;
#define FIELD(name) h.name = config[#name].cast<decltype(h.name)>()
  FIELD(sram_bytes);
  FIELD(macs_per_cycle);
  FIELD(elements_per_cycle);
  FIELD(dma_bytes_per_cycle);
  FIELD(dma_setup_cycles);
  FIELD(compute_setup_cycles);
  FIELD(pim_macs_per_cycle);
  FIELD(pim_bytes_per_cycle);
  FIELD(clock_hz);
#undef FIELD
  return h;
}
py::dict metrics(const Result &r) {
  py::dict out;
#define OUTPUT(name) out[#name] = r.name
  OUTPUT(cycles);
  OUTPUT(latency_ns);
  OUTPUT(dram_read_bytes);
  OUTPUT(dram_write_bytes);
  OUTPUT(offchip_read_bytes);
  OUTPUT(pim_local_read_bytes);
  OUTPUT(peak_sram_bytes);
  OUTPUT(compute_busy);
  OUTPUT(dma_busy);
  OUTPUT(pim_busy);
  OUTPUT(dram_busy);
  OUTPUT(bottleneck);
#undef OUTPUT
  double denom = r.cycles ? static_cast<double>(r.cycles) : 1.0;
  out["compute_utilization"] = r.compute_busy / denom;
  out["dma_utilization"] = r.dma_busy / denom;
  out["pim_utilization"] = r.pim_busy / denom;
  out["dram_utilization"] = r.dram_busy / denom;
  out["offchip_bytes"] = py::int_(r.offchip_read_bytes) + py::int_(r.dram_write_bytes);
  py::list trace;
  for (const auto &e : r.trace) {
    py::dict d;
    d["id"] = e.id;
    d["op"] = e.op;
    d["start"] = e.start;
    d["end"] = e.end;
    trace.append(d);
  }
  out["trace"] = trace;
  return out;
}
py::dict decoded(const ImageInstruction &i) {
  py::dict d;
  const auto &t = i.timing;
#define T(name) d[#name] = t.name
  T(id);
  T(op);
  T(deps);
  T(buffer);
  T(sram_bytes);
  T(working_set_bytes);
  T(bytes);
  T(macs);
  T(elements);
  T(local_read_bytes);
#undef T
  d["flags"] = i.flags;
  d["mode"] = i.mode;
  const std::array<const char *, 4> ids{"tensor_a", "tensor_b", "tensor_c", "tensor_out"};
  const std::array<const char *, 4> addresses{"a", "b", "c", "out"};
  const std::array<const char *, 6> dimensions{"m0", "n0", "k0", "m", "n", "k"};
  for (unsigned j = 0; j < 4; ++j) {
    d[ids[j]] = i.tensors[j];
    d[addresses[j]] = i.addresses[j];
  }
  for (unsigned j = 0; j < 6; ++j)
    d[dimensions[j]] = i.dims[j];
  d["alpha"] = i.alpha;
  d["beta"] = i.beta;
  d["flat_start"] = i.flat_start;
  d["count"] = i.count;
  return d;
}
} // namespace
PYBIND11_MODULE(_native, m) {
  m.doc() = "C++17 NPU scheduler, binary decoder and addressed FP32 interpreter";
  m.def("simulate", [](const py::list &program, const py::dict &config) {
    auto h = hardware(config);
    std::vector<Instruction> instructions;
    for (const auto &item : program) {
      auto d = py::cast<py::dict>(item);
      Instruction i;
      i.id = d["id"].cast<int>();
      i.op = d["op"].cast<std::string>();
      i.deps = d["deps"].cast<std::vector<int>>();
#define OPTIONAL(name)                                                                             \
  if (d.contains(#name))                                                                           \
  i.name = d[#name].cast<decltype(i.name)>()
      OPTIONAL(buffer);
      OPTIONAL(sram_bytes);
      OPTIONAL(working_set_bytes);
      OPTIONAL(bytes);
      OPTIONAL(macs);
      OPTIONAL(elements);
      OPTIONAL(local_read_bytes);
#undef OPTIONAL
      instructions.push_back(std::move(i));
    }
    Result r;
    {
      py::gil_scoped_release release;
      r = simulate(instructions, h);
    }
    return metrics(r);
  });
  m.def("decode_image", [](const py::bytes &data) {
    auto bytes = data.cast<std::string>();
    std::vector<ImageInstruction> instructions;
    {
      py::gil_scoped_release release;
      instructions = decode_image(bytes);
    }
    py::list out;
    for (const auto &i : instructions)
      out.append(decoded(i));
    return out;
  });
  m.def("execute_image",
        [](const py::bytes &data, const py::bytes &weights, const py::list &descriptors,
           const py::dict &input, U64 dram_bytes, const py::dict &config) {
          auto h = hardware(config);
          auto bytes = data.cast<std::string>(), w = weights.cast<std::string>();
          std::vector<TensorImage> tensors;
          for (const auto &item : descriptors) {
            auto d = py::cast<py::dict>(item);
            TensorImage t;
            t.name = d["id"].cast<std::string>();
#define FIELD(name) t.name = d[#name].cast<decltype(t.name)>()
            FIELD(address);
            FIELD(size);
            FIELD(allocation_size);
            FIELD(shape);
            FIELD(role_bits);
            FIELD(first);
            FIELD(last);
#undef FIELD
            tensors.push_back(std::move(t));
          }
          std::map<unsigned, std::string> inputs;
          for (auto item : input)
            inputs[item.first.cast<unsigned>()] = item.second.cast<std::string>();
          ImageResult r;
          {
            py::gil_scoped_release release;
            r = execute_image(bytes, w, tensors, inputs, dram_bytes, h);
          }
          py::dict out, values;
          for (const auto &[id, value] : r.outputs)
            values[py::int_(id)] = py::bytes(value);
          out["outputs"] = values;
          out["metrics"] = metrics(r.timing);
          return out;
        });
}
