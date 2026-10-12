// nevpt2.gradsrc:archive -- the bytes of a `.npz`: NPY v1.0 members in a
// stored (uncompressed) zip archive, built and parsed in memory. An internal
// partition: nothing here is exported, so gradsrc.cpp reaches it (`import
// :archive;`) and no importer of nevpt2.gradsrc does.
//
// Only what writeGradSrc needs: '<f8' arrays of any rank and 0-d '<i8'
// scalars, C order, no compression, no zip64 (every member and the whole
// archive stay under 4 GiB, which a `.gradsrc` -- an n_mo^2 matrix and two CI
// vectors -- is by orders of magnitude; a larger one is refused, not
// truncated). The parser accepts exactly that layout back.
module;

#include "error_handling/error_macros.h"  // NEVPT2_TRY: a macro, which no import carries

module nevpt2.gradsrc:archive;

import std;
import nevpt2.gradsrc;

namespace nevpt2 {

// Every field below is written and read byte by byte, least significant
// first, so the layout does not depend on the host; the doubles go through
// std::bit_cast, which assumes the host's double is IEEE-754 binary64 -- true
// of every host this builds for.
static_assert(std::numeric_limits<double>::is_iec559,
              "a .npy '<f8' member is IEEE-754 binary64");

// One member: a name (without ".npy") and either float data of any shape or a
// 0-d integer.
struct NpyMember {
  std::string name;
  bool isInt = false;
  std::vector<int64_t> shape;  // empty: 0-d
  std::vector<double> f64;     // !isInt: C-order data
  int64_t i64 = 0;             // isInt: the scalar
};

// --- CRC-32 (ISO 3309 / zip's), the reflected 0xEDB88320 polynomial ---------

constexpr std::array<std::uint32_t, 256> makeCrcTable() {
  std::array<std::uint32_t, 256> t{};
  for (std::uint32_t n = 0; n < 256; ++n) {
    std::uint32_t c = n;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    t[n] = c;
  }
  return t;
}
constexpr std::array<std::uint32_t, 256> kCrcTable = makeCrcTable();

std::uint32_t crc32Of(const std::span<const std::uint8_t> bytes) {
  std::uint32_t c = 0xFFFFFFFFu;
  for (const std::uint8_t b : bytes) c = kCrcTable[(c ^ b) & 0xFFu] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

// --- little-endian byte appends and reads -----------------------------------

void putLe(std::vector<std::uint8_t>& out, const std::uint64_t v, const int nbytes) {
  for (int i = 0; i < nbytes; ++i) out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}

std::uint64_t getLe(const std::span<const std::uint8_t> in, const std::size_t at,
                    const int nbytes) {
  std::uint64_t v = 0;
  for (int i = 0; i < nbytes; ++i) v |= static_cast<std::uint64_t>(in[at + i]) << (8 * i);
  return v;
}

// --- NPY v1.0 ---------------------------------------------------------------

constexpr std::string_view kNpyMagic = "\x93NUMPY";

// NumPy's own shape spelling: "()", "(n,)", "(a, b)" -- what
// numpy.lib.format writes and ast.literal_eval reads back.
std::string shapeTuple(const std::vector<int64_t>& shape) {
  if (shape.empty()) return "()";
  std::string s = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  if (shape.size() == 1) s += ",";
  return s + ")";
}

// One member's whole .npy file. The header is padded with spaces to a '\n' so
// that magic + version + length + header is a multiple of 64 bytes, as
// numpy.lib.format requires of the data offset.
std::vector<std::uint8_t> npyBytes(const NpyMember& m) {
  std::string header = std::format("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
                                   m.isInt ? "<i8" : "<f8", shapeTuple(m.shape));
  const std::size_t fixed = kNpyMagic.size() + 2 + 2;
  const std::size_t total = (fixed + header.size() + 1 + 63) / 64 * 64;
  header.append(total - fixed - header.size() - 1, ' ');
  header += '\n';
  std::vector<std::uint8_t> out(kNpyMagic.begin(), kNpyMagic.end());
  out.push_back(1);  // major version
  out.push_back(0);  // minor version
  putLe(out, header.size(), 2);
  out.insert(out.end(), header.begin(), header.end());
  if (m.isInt) {
    putLe(out, static_cast<std::uint64_t>(m.i64), 8);
  } else {
    for (const double d : m.f64) putLe(out, std::bit_cast<std::uint64_t>(d), 8);
  }
  return out;
}

// The value of `key` in an NPY header dict, as the text between its ": " and
// the next ", '" or the closing "}" -- enough for the three keys and the one
// spelling npyBytes writes (and numpy.lib.format writes the same way).
Result<std::string_view> headerField(const std::string_view header, const std::string_view key,
                                     const std::string_view where) {
  const std::string needle = std::format("'{}': ", key);
  const std::size_t at = header.find(needle);
  if (at == std::string_view::npos)
    return err_io(std::format("{}: NPY header has no '{}': {}", where, key, header));
  const std::size_t from = at + needle.size();
  std::size_t to = from;
  if (from < header.size() && header[from] == '(') {
    to = header.find(')', from);
    if (to == std::string_view::npos)
      return err_io(std::format("{}: NPY header's shape is unterminated", where));
    ++to;
  } else {
    to = header.find(',', from);
    if (to == std::string_view::npos) to = header.find('}', from);
    if (to == std::string_view::npos)
      return err_io(std::format("{}: NPY header's '{}' is unterminated", where, key));
  }
  return header.substr(from, to - from);
}

Result<std::vector<int64_t>> parseShape(std::string_view tuple, const std::string_view where) {
  if (tuple.size() < 2 || tuple.front() != '(' || tuple.back() != ')')
    return err_io(std::format("{}: NPY shape is not a tuple: {}", where, tuple));
  tuple = tuple.substr(1, tuple.size() - 2);
  std::vector<int64_t> shape;
  while (!tuple.empty()) {
    while (!tuple.empty() && (tuple.front() == ' ' || tuple.front() == ',')) tuple.remove_prefix(1);
    if (tuple.empty()) break;
    int64_t v = 0;
    const auto [end, ec] = std::from_chars(tuple.data(), tuple.data() + tuple.size(), v);
    if (ec != std::errc{} || v < 0)
      return err_io(std::format("{}: NPY shape has a bad extent: {}", where, tuple));
    shape.push_back(v);
    tuple.remove_prefix(static_cast<std::size_t>(end - tuple.data()));
  }
  return shape;
}

// One .npy file's bytes back into a member named `name`.
Result<NpyMember> parseNpy(const std::span<const std::uint8_t> b, std::string name,
                           const std::string_view where) {
  const std::size_t fixed = kNpyMagic.size() + 2 + 2;
  if (b.size() < fixed ||
      !std::equal(kNpyMagic.begin(), kNpyMagic.end(), b.begin(),
                  [](char c, std::uint8_t u) { return static_cast<std::uint8_t>(c) == u; }))
    return err_io(std::format("{}: member {} is not an NPY file", where, name));
  if (b[6] != 1 || b[7] != 0)
    return err_io(std::format("{}: member {} is NPY version {}.{}, not 1.0", where, name,
                              int{b[6]}, int{b[7]}));
  const std::size_t hlen = getLe(b, 8, 2);
  if (fixed + hlen > b.size())
    return err_io(std::format("{}: member {}'s NPY header runs past its end", where, name));
  const std::string header(reinterpret_cast<const char*>(b.data()) + fixed, hlen);
  const std::string_view descr = NEVPT2_TRY(headerField(header, "descr", where));
  const std::string_view order = NEVPT2_TRY(headerField(header, "fortran_order", where));
  const std::string_view tuple = NEVPT2_TRY(headerField(header, "shape", where));
  if (order != "False")
    return err_io(std::format("{}: member {} is Fortran-ordered", where, name));
  NpyMember m;
  m.name = std::move(name);
  m.shape = NEVPT2_TRY(parseShape(tuple, where));
  std::size_t count = 1;
  for (const int64_t e : m.shape) count *= static_cast<std::size_t>(e);
  const std::size_t dataAt = fixed + hlen;
  if (b.size() - dataAt != 8 * count)
    return err_io(std::format("{}: member {} holds {} data bytes for {} elements", where,
                              m.name, b.size() - dataAt, count));
  if (descr == "'<i8'") {
    if (!m.shape.empty())
      return err_io(std::format("{}: member {} is an integer array, not a 0-d scalar", where,
                                m.name));
    m.isInt = true;
    m.i64 = static_cast<int64_t>(getLe(b, dataAt, 8));
  } else if (descr == "'<f8'") {
    m.f64.resize(count);
    for (std::size_t i = 0; i < count; ++i)
      m.f64[i] = std::bit_cast<double>(getLe(b, dataAt + 8 * i, 8));
  } else {
    return err_io(std::format("{}: member {} has dtype {}, not '<f8' or '<i8'", where, m.name,
                              descr));
  }
  return m;
}

// --- the zip container --------------------------------------------------------

constexpr std::uint32_t kLocalSig = 0x04034b50u;
constexpr std::uint32_t kCentralSig = 0x02014b50u;
constexpr std::uint32_t kEndSig = 0x06054b50u;
constexpr std::size_t kLocalFixed = 30;
constexpr std::size_t kCentralFixed = 46;
constexpr std::size_t kEndFixed = 22;
// "version needed to extract" 2.0, and a fixed DOS timestamp (1980-01-01
// 00:00): the file is then a function of its contents alone.
constexpr std::uint16_t kZipVersion = 20;
constexpr std::uint16_t kDosDate = (0 << 9) | (1 << 5) | 1;
constexpr std::uint16_t kDosTime = 0;

// The whole archive. An archive that would need zip64 is refused (IO Error):
// see the header comment.
Result<std::vector<std::uint8_t>> npzBytes(const std::vector<NpyMember>& members,
                                           const std::string_view where) {
  constexpr std::uint64_t kMax = 0xFFFFFFFFu;
  if (members.size() > 0xFFFFu)
    return err_io(std::format("{}: {} members is past a zip archive's limit", where,
                              members.size()));
  std::vector<std::uint8_t> out;
  std::vector<std::uint8_t> central;
  for (const NpyMember& m : members) {
    const std::vector<std::uint8_t> data = npyBytes(m);
    const std::string file = m.name + ".npy";
    if (data.size() > kMax || out.size() > kMax)
      return err_io(std::format("{}: member {} would need zip64", where, file));
    const std::uint32_t crc = crc32Of(data);
    const std::uint64_t offset = out.size();
    const auto header = [&](std::vector<std::uint8_t>& h, const bool isCentral) {
      putLe(h, isCentral ? kCentralSig : kLocalSig, 4);
      if (isCentral) putLe(h, kZipVersion, 2);  // version made by
      putLe(h, kZipVersion, 2);                 // version needed
      putLe(h, 0, 2);                           // flags
      putLe(h, 0, 2);                           // method: stored
      putLe(h, kDosTime, 2);
      putLe(h, kDosDate, 2);
      putLe(h, crc, 4);
      putLe(h, data.size(), 4);  // compressed size
      putLe(h, data.size(), 4);  // uncompressed size
      putLe(h, file.size(), 2);
      putLe(h, 0, 2);  // extra field length
      if (isCentral) {
        putLe(h, 0, 2);  // comment length
        putLe(h, 0, 2);  // disk number start
        putLe(h, 0, 2);  // internal attributes
        putLe(h, 0, 4);  // external attributes
        putLe(h, offset, 4);
      }
      h.insert(h.end(), file.begin(), file.end());
    };
    header(out, false);
    out.insert(out.end(), data.begin(), data.end());
    header(central, true);
  }
  const std::uint64_t centralAt = out.size();
  if (centralAt > kMax || central.size() > kMax)
    return err_io(std::format("{}: the archive would need zip64", where));
  out.insert(out.end(), central.begin(), central.end());
  putLe(out, kEndSig, 4);
  putLe(out, 0, 2);  // this disk
  putLe(out, 0, 2);  // the central directory's disk
  putLe(out, members.size(), 2);
  putLe(out, members.size(), 2);
  putLe(out, central.size(), 4);
  putLe(out, centralAt, 4);
  putLe(out, 0, 2);  // comment length
  return out;
}

// The archive's members, read through its central directory and each local
// header, every CRC checked. Only what npzBytes writes is accepted: no archive
// comment, stored members, no data descriptors.
Result<std::vector<NpyMember>> parseNpz(const std::span<const std::uint8_t> b,
                                        const std::string_view where) {
  if (b.size() < kEndFixed || getLe(b, b.size() - kEndFixed, 4) != kEndSig)
    return err_io(std::format("{}: no zip end-of-central-directory record at its end", where));
  const std::size_t end = b.size() - kEndFixed;
  const std::size_t entries = getLe(b, end + 10, 2);
  const std::size_t centralSize = getLe(b, end + 12, 4);
  const std::size_t centralAt = getLe(b, end + 16, 4);
  if (centralAt + centralSize != end)
    return err_io(std::format("{}: the zip central directory does not end where its record "
                              "says",
                              where));
  std::vector<NpyMember> members;
  std::size_t at = centralAt;
  for (std::size_t e = 0; e < entries; ++e) {
    if (at + kCentralFixed > end || getLe(b, at, 4) != kCentralSig)
      return err_io(std::format("{}: zip central directory entry {} is corrupt", where, e));
    const std::size_t flags = getLe(b, at + 8, 2);
    const std::size_t method = getLe(b, at + 10, 2);
    const std::uint32_t crc = static_cast<std::uint32_t>(getLe(b, at + 16, 4));
    const std::size_t csize = getLe(b, at + 20, 4);
    const std::size_t usize = getLe(b, at + 24, 4);
    const std::size_t nlen = getLe(b, at + 28, 2);
    const std::size_t xlen = getLe(b, at + 30, 2);
    const std::size_t clen = getLe(b, at + 32, 2);
    const std::size_t local = getLe(b, at + 42, 4);
    if (at + kCentralFixed + nlen > end)
      return err_io(std::format("{}: zip central directory entry {} runs past it", where, e));
    std::string file(reinterpret_cast<const char*>(b.data()) + at + kCentralFixed, nlen);
    if (method != 0 || flags != 0 || csize != usize)
      return err_io(std::format("{}: member {} is not stored uncompressed", where, file));
    if (local + kLocalFixed > centralAt || getLe(b, local, 4) != kLocalSig)
      return err_io(std::format("{}: member {}'s local header is corrupt", where, file));
    const std::size_t dataAt =
        local + kLocalFixed + getLe(b, local + 26, 2) + getLe(b, local + 28, 2);
    if (dataAt + csize > centralAt)
      return err_io(std::format("{}: member {} runs past the archive's data", where, file));
    const std::span<const std::uint8_t> data = b.subspan(dataAt, csize);
    if (crc32Of(data) != crc)
      return err_io(std::format("{}: member {} fails its CRC-32", where, file));
    if (!file.ends_with(".npy"))
      return err_io(std::format("{}: member {} is not a .npy file", where, file));
    file.resize(file.size() - 4);
    members.push_back(NEVPT2_TRY(parseNpy(data, std::move(file), where)));
    at += kCentralFixed + nlen + xlen + clen;
  }
  return members;
}

}  // namespace nevpt2
