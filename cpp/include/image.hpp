#pragma once
#include "simulator.hpp"
#include <array>
#include <map>

namespace npuforge {
constexpr std::uint32_t NoTensor = 0xffffffffU;
struct ImageInstruction {
  Instruction timing;
  std::uint32_t flags = 0, mode = 0;
  std::array<std::uint32_t, 4> tensors{NoTensor, NoTensor, NoTensor, NoTensor};
  std::array<U64, 4> addresses{};
  U64 flat_start = 0, count = 0;
  std::array<std::uint32_t, 6> dims{}; // m0,n0,k0,m,n,k
  float alpha = 1, beta = 1;
};
struct TensorImage {
  std::string name;
  U64 address = 0, size = 0, allocation_size = 0, first = 0, last = 0;
  unsigned role_bits = 0;
  std::vector<U64> shape;
};
struct ImageResult {
  Result timing;
  std::map<unsigned, std::string> outputs;
};
std::vector<ImageInstruction> decode_image(const std::string &bytes);
ImageResult execute_image(const std::string &bytes, const std::string &weights,
                          const std::vector<TensorImage> &tensors,
                          const std::map<unsigned, std::string> &inputs, U64 dram_bytes,
                          const Hardware &hardware);
} // namespace npuforge
