#!/usr/bin/env bash
# 激活 CFU-Playground 的 conda 环境（无 FPGA / Renode 仿真用）
# 用法: source /home/hjadmin/zw/infer-infrastructure/cfu-playground/enter-env.sh
CFU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${CFU_ROOT}/env/conda/bin/activate" cfu-common
# shellcheck disable=SC1091
source "${CFU_ROOT}/environment"
export PATH="${CFU_ROOT}/third_party/renode:${PATH}"
echo "cfu-common 已激活: $(which python)"
echo "RISC-V GCC: $(which riscv32-elf-newlib-gcc)"
echo "Renode: $(which renode)"
echo "下一步示例:"
echo "  cd ${CFU_ROOT}/proj/example_cfu && make renode-headless"
echo "  或: cd ${CFU_ROOT}/proj/example_cfu && make renode-test"
