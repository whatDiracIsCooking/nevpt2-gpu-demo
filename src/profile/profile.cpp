// The body of nevpt2.profile -- see the interface.
module;

#include <cstdio>  // stdout: a macro, which `import std` does not carry

module nevpt2.profile;

import std;
import :spans;

namespace nevpt2::profile {

std::size_t begin(std::string_view label, wwrStream_t stream, int64_t outSize,
                  int64_t contractedSize) {
  // Raw events, not WarpWraps' EventWrapper: its default flags disable
  // timing (https://github.com/whatDiracIsCooking/WarpWraps/issues/315), and
  // wwrEventDefault is what makes the pair timeable.
  Span s{g_section, std::string(label), nullptr, nullptr, outSize, contractedSize};
  gpuCheck(wwrEventCreateWithFlags(&s.start, wwrEventDefault));
  gpuCheck(wwrEventCreateWithFlags(&s.stop, wwrEventDefault));
  gpuCheck(wwrEventRecord(s.start, stream));
  g_spans.push_back(std::move(s));
  return g_spans.size() - 1;
}

void end(std::size_t span, wwrStream_t stream) {
  gpuCheck(wwrEventRecord(g_spans[span].stop, stream));
}

void setEnabled(bool enabled) { g_enabled = enabled; }

Section::Section(std::string_view name) : previous_(std::exchange(g_section, name)) {}
Section::~Section() { g_section = std::move(previous_); }

std::vector<ProfileRow> report(std::string_view section) {
  std::vector<ProfileRow> rows;
  std::vector<Span> kept;
  for (auto& s : g_spans) {
    if (s.section != section) {
      kept.push_back(std::move(s));
      continue;
    }
    float ms = 0.0f;
    gpuCheck(wwrEventElapsedTime(&ms, s.start, s.stop));
    (void)wwrEventDestroy(s.start);
    (void)wwrEventDestroy(s.stop);
    auto it = std::ranges::find(rows, s.label, &ProfileRow::label);
    if (it == rows.end()) it = rows.insert(rows.end(), ProfileRow{s.label});
    it->calls += 1;
    it->totalMs += ms;
    it->outSize = s.outSize;
    it->contractedSize = s.contractedSize;
  }
  g_spans = std::move(kept);

  for (auto& row : rows) row.meanUs = row.totalMs * 1000.0 / row.calls;
  std::ranges::sort(rows, std::ranges::greater{}, &ProfileRow::totalMs);
  return rows;
}

void printReport(std::string_view section, double wallSeconds) {
  if (!enabled()) return;
  const std::vector<ProfileRow> rows = report(section);
  double sumMs = 0.0;
  for (const auto& row : rows) sumMs += row.totalMs;
  const std::string name(section);
  if (wallSeconds > 0.0) {
    std::printf(
        "\n--- %s profile (%zu labels, %.2fms summed GPU time of a %.2fs stage; the rest is "
        "uploads, allocations, host work and event overhead) ---\n",
        name.c_str(), rows.size(), sumMs, wallSeconds);
  } else {
    std::printf("\n--- %s profile (%zu labels, %.2fms summed GPU time) ---\n", name.c_str(),
                rows.size(), sumMs);
  }
  std::printf("%-40s %6s %11s %11s %7s %12s %12s\n", "label", "calls", "total(ms)",
              "mean(us)", "%", "outSize", "contracted");
  auto size = [](int64_t n) { return n > 0 ? std::to_string(n) : std::string("-"); };
  for (const auto& row : rows) {
    std::printf("%-40s %6d %11.3f %11.2f %6.1f%% %12s %12s\n", row.label.c_str(), row.calls,
                row.totalMs, row.meanUs, sumMs > 0.0 ? 100.0 * row.totalMs / sumMs : 0.0,
                size(row.outSize).c_str(), size(row.contractedSize).c_str());
  }
}

}  // namespace nevpt2::profile
