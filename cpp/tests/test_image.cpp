#include "image.hpp"
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace npuforge;
namespace {
void require(bool test, const char *message) {
  if (!test)
    throw std::runtime_error(message);
}
void rejects(const std::function<void()> &fn) {
  bool caught = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  require(caught, "malformed image was accepted");
}
void put(std::string &s, U64 offset, U64 value, unsigned bytes = 4) {
  for (unsigned j = 0; j < bytes; ++j)
    s.at(offset + j) = static_cast<char>((value >> (j * 8)) & 255);
}
std::string program(bool pim = false, bool missing_dependency = false) {
  std::vector<unsigned> ops =
      pim ? std::vector<unsigned>{1, 3, 9, 4, 2} : std::vector<unsigned>{1, 3, 3, 5, 4, 2};
  std::vector<std::vector<unsigned>> deps =
      pim ? std::vector<std::vector<unsigned>>{{}, {0}, {1}, {2}, {3}}
          : std::vector<std::vector<unsigned>>{
                {},  {0},
                {0}, missing_dependency ? std::vector<unsigned>{1} : std::vector<unsigned>{1, 2},
                {3}, {4}};
  U64 count = 0;
  for (const auto &d : deps)
    count += d.size();
  std::string s(32 + ops.size() * 256 + count * 4, '\0');
  s.replace(0, 8, std::string("NPUFIMG\0", 8));
  put(s, 8, 1, 2);
  put(s, 10, 256, 2);
  put(s, 12, 1);
  put(s, 16, ops.size(), 8);
  put(s, 24, count, 8);
  U64 pool = 0;
  for (unsigned id = 0; id < ops.size(); ++id) {
    U64 at = 32 + 256 * id;
    unsigned op = ops[id];
    put(s, at, op);
    put(s, at + 4, id);
    put(s, at + 20, pool);
    put(s, at + 24, deps[id].size());
    put(s, at + 28, 1);
    for (unsigned dep : deps[id]) {
      put(s, 32 + ops.size() * 256 + pool * 4, dep);
      ++pool;
    }
    for (unsigned f = 0; f < 4; ++f)
      put(s, at + 32 + 4 * f, NoTensor);
    put(s, at + 168, 0x3f800000);
    put(s, at + 172, 0x3f800000);
    if (op == 1) {
      put(s, at + 80, pim ? 8 : 12, 8);
      continue;
    }
    if (op == 2)
      continue;
    for (unsigned f = 3; f < 6; ++f)
      put(s, at + 144 + 4 * f, 1);
    if (op == 3) {
      bool b = !pim && id == 2;
      put(s, at + 12, b ? 1 : 0);
      put(s, at + 32, b ? 1 : 0);
      put(s, at + 44, 2);
      put(s, at + 48, b ? 0 : 64, 8);
      put(s, at + 72, b ? 4 : 0, 8);
      put(s, at + 96, 4, 8);
    } else if (op == 4) {
      put(s, at + 12, 4);
      put(s, at + 44, 2);
      put(s, at + 48, pim ? 4 : 8, 8);
      put(s, at + 72, 128, 8);
      put(s, at + 96, 4, 8);
    } else {
      put(s, at + 32, 0);
      put(s, at + 36, 1);
      put(s, at + 44, 2);
      put(s, at + 56, pim ? 0 : 4, 8);
      put(s, at + 72, pim ? 4 : 8, 8);
      put(s, at + 104, 1, 8);
      put(s, at + 88, pim ? 8 : 12, 8);
      if (pim)
        put(s, at + 120, 4, 8);
    }
  }
  return s;
}
std::vector<TensorImage> tensors() {
  return {{"a", 64, 4, 64, 0, 1, 2, {1, 1}},
          {"b", 0, 4, 64, 0, 1, 1, {1, 1}},
          {"out", 128, 4, 64, 0, 1, 4, {1, 1}}};
}
} // namespace
int main() {
  try {
    Hardware h;
    std::string weights(64, '\0');
    put(weights, 0, 0x40400000); // 3.0
    std::string input(4, '\0');
    put(input, 0, 0x40000000); // 2.0
    for (bool pim : {false, true}) {
      auto r = execute_image(program(pim), weights, tensors(), {{0, input}}, 192, h);
      require(r.outputs.at(2) == std::string("\0\0\xc0\x40", 4),
              "native binary matrix output must equal 6");
      require(r.timing.cycles == (pim ? 24U : 32U), "binary resource timing");
    }
    auto valid = program();
    for (std::size_t n : {0U, 7U, 31U, 32U, 100U})
      rejects([&] { decode_image(valid.substr(0, n)); });
    for (unsigned offset : {0U, 8U, 10U, 12U, 32U, 60U, 208U}) {
      auto bad = valid;
      bad[offset] = static_cast<char>(255);
      rejects([&] { decode_image(bad); });
    }
    rejects([&] { execute_image(program(false, true), weights, tensors(), {{0, input}}, 192, h); });
    rejects([&] { execute_image(valid, weights, tensors(), {}, 192, h); });
    auto bad = valid;
    put(bad, 32 + 3 * 256 + 72, 12, 8);
    rejects([&] { execute_image(bad, weights, tensors(), {{0, input}}, 192, h); });
    auto overlap = tensors();
    overlap[2].address = 64;
    rejects([&] { execute_image(valid, weights, overlap, {{0, input}}, 192, h); });
    std::cout << "Binary decoding, native arithmetic, timing and malformed images passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
