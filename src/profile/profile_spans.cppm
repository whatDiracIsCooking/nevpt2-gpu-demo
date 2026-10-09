// nevpt2.profile:spans -- the recorder's state: every recorded Span (an event
// pair and its labels) and the innermost open Section's name.
// An internal partition: nothing here is exported, so it is reachable from the
// units that `import :spans;` and from no importer of nevpt2.profile.
// (Used by profile.cpp.)
module nevpt2.profile:spans;

import std;
import nevpt2.profile;

namespace nevpt2::profile {

struct Span {
  std::string section;
  std::string label;
  wwrEvent_t start = nullptr;
  wwrEvent_t stop = nullptr;
  int64_t outSize = 0;
  int64_t contractedSize = 0;
};

std::vector<Span> g_spans;

std::string g_section;  // the innermost open Section's name

}  // namespace nevpt2::profile
