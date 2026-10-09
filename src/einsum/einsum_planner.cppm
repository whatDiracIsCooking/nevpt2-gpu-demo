// nevpt2.einsum:planner -- subscript parsing: an einsum string checked against
// its operands and recorded as the EinsumPlan POD the kernel reads, and
// the subscripts a transpose is issued as.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :planner;` and from no importer of nevpt2.einsum.
// (Used by einsum.cpp.)
module;

// The one POD the plan is (einsum_plan.h), textually in the GMF.
#include "einsum/einsum_plan.h"

module nevpt2.einsum:planner;

import std;
import nevpt2.einsum;

namespace nevpt2 {

std::vector<std::string> splitInputLabels(const std::string& lhs) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : lhs) {
    if (c == ',') {
      out.push_back(cur);
      cur.clear();
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      cur.push_back(c);
    }
  }
  out.push_back(cur);
  return out;
}

// Builds the EinsumPlan and the output shape for `subscripts` against
// `operands` -- the label bookkeeping a host einsum does (parse, check every
// label's dimension is consistent everywhere it appears, classify output vs.
// contracted; tensor/tensor.cpp had one, since removed), just recorded into a
// POD the kernel reads instead of driving an immediate host-side loop.
// Every subscript string is a literal at its call site, so a malformed one --
// or one that does not fit its operands -- is our bug, never bad input: each
// failure below goes through check() (the abort tier), naming `loc`, the
// deviceEinsum* caller's file:line. The message is only formatted on failure.
EinsumPlan buildPlan(const std::string& subscripts,
                      const std::vector<const DeviceTensor*>& operands,
                      std::vector<int64_t>* outDimsOut, const std::source_location& loc) {
  auto arrow = subscripts.find("->");
  if (arrow == std::string::npos)
    check(false, std::format("deviceEinsum: subscripts must use '->': {}", subscripts), loc);
  std::string lhs = subscripts.substr(0, arrow);
  std::string rhs = subscripts.substr(arrow + 2);
  std::vector<std::string> opLabels = splitInputLabels(lhs);
  if (opLabels.size() != operands.size())
    check(false,
          std::format("deviceEinsum: {} operand label-groups but {} operands given: {}",
                      opLabels.size(), operands.size(), subscripts),
          loc);
  if (std::ssize(operands) > kEinsumMaxOperands)
    check(false, std::format("deviceEinsum: too many operands: {}", subscripts), loc);

  std::unordered_map<char, int64_t> labelDim;
  for (std::size_t o = 0; o < operands.size(); ++o) {
    const std::string& ls = opLabels[o];
    if (std::ssize(ls) != operands[o]->rank())
      check(false,
            std::format("deviceEinsum: operand {} rank {} != label length {} ({})", o,
                        operands[o]->rank(), ls.size(), subscripts),
            loc);
    if (operands[o]->rank() > kEinsumMaxOperandRank)
      check(false,
            std::format("deviceEinsum: operand rank exceeds kEinsumMaxOperandRank: {}",
                        subscripts),
            loc);
    for (std::size_t a = 0; a < ls.size(); ++a) {
      char c = ls[a];
      int64_t d = operands[o]->dims[a];
      auto it = labelDim.find(c);
      if (it == labelDim.end()) {
        labelDim.emplace(c, d);
      } else if (it->second != d) {
        check(false,
              std::format("deviceEinsum: inconsistent dim for label '{}' in {}", c, subscripts),
              loc);
      }
    }
  }
  for (char c : rhs) {
    if (labelDim.find(c) == labelDim.end())
      check(false,
            std::format("deviceEinsum: output label '{}' not in any operand ({})", c,
                        subscripts),
            loc);
  }

  std::string allLabels = rhs;
  for (const auto& kv : labelDim) {
    if (allLabels.find(kv.first) == std::string::npos) allLabels.push_back(kv.first);
  }
  const int64_t nContracted = std::ssize(allLabels) - std::ssize(rhs);
  if (nContracted > kEinsumMaxContracted || std::ssize(allLabels) > kEinsumMaxLabels)
    check(false,
          std::format("deviceEinsum: too many distinct/contracted labels: {}", subscripts), loc);

  // The plan is a POD the kernel reads (einsum/einsum_plan.h): its counts are
  // int, its extents and strides int64_t. The counts are bounded by the
  // kEinsumMax* checks above; each is narrowed into the plan once, here.
  EinsumPlan plan;
  plan.nOperands = narrowTo<int>(std::ssize(operands));
  plan.outRank = narrowTo<int>(std::ssize(rhs));
  plan.nContractedLabels = narrowTo<int>(nContracted);

  outDimsOut->clear();
  plan.outSize = 1;
  // Every label looked up below is a key of labelDim (each output label was
  // checked above; the contracted ones were taken from its keys), so
  // operator[] only ever finds, never inserts.
  for (int i = 0; i < plan.outRank; ++i) {
    int64_t d = labelDim[rhs[i]];
    plan.outDims[i] = d;
    outDimsOut->push_back(d);
    plan.outSize *= d;
  }
  plan.contractedSize = 1;
  for (int i = 0; i < plan.nContractedLabels; ++i) {
    int64_t d = labelDim[allLabels[plan.outRank + i]];
    plan.contractedDims[i] = d;
    plan.contractedSize *= d;
  }

  for (std::size_t o = 0; o < operands.size(); ++o) {
    plan.opRank[o] = narrowTo<int>(operands[o]->rank());
    for (int a = 0; a < plan.opRank[o]; ++a) {
      char c = opLabels[o][a];
      // Every label is in allLabels (built from them above), so the search
      // never runs off its end.
      plan.opAxisLabel[o][a] = narrowTo<int>(std::ranges::find(allLabels, c) - allLabels.begin());
      plan.opAxisStride[o][a] = operands[o]->strides[a];
    }
  }
  return plan;
}

// "ABCDEFGH"[:rank] labels x's own axes; the transpose's output labels are
// those same letters read in `axes` order (out label at position i = x's
// label at axis axes[i]) -- numpy's `X.transpose(axes)` convention. Verified,
// when this was ported, against tensor/tensor.cpp's independently-implemented
// host `transpose()` on the same {1,4,0,2,5,3} permutation this port's biggest
// user (make_a16/make_a22) needs, in a one-off smoke test no longer in the
// tree; the golden check now covers it through every class energy.
std::string transposeSubscripts(int64_t rank, const std::vector<int>& axes) {
  static const char kAlphabet[] = "ABCDEFGH";
  std::string in(kAlphabet, kAlphabet + rank);
  std::string out(in.size(), '?');
  for (int64_t i = 0; i < rank; ++i) out[i] = kAlphabet[axes[i]];
  return in + "->" + out;
}

}  // namespace nevpt2
