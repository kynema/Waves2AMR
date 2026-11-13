#include "read_modes.h"
#include "gtest/gtest.h"
#include <filesystem>
#include <iomanip>
#include <unistd.h>

namespace {
class ModesIO : public testing::Test
{
protected:
    std::string filename;
    void SetUp() override
    {
        auto pattern =
            (std::filesystem::temp_directory_path() / "w2a-modes-XXXXXX")
                .string();
        const int fd = mkstemp(pattern.data());
        ASSERT_GE(fd, 0);
        close(fd);
        filename = pattern;
    }
    void TearDown() override { std::filesystem::remove(filename); }

    void write(bool ocean, bool bad_derivative = false)
    {
        std::ofstream file(filename, std::ios::binary);
        const int width = ocean ? 9 : 16;
        const int components = ocean ? 2 : 1;
        const int slots = ocean ? 12 * width * 2 : 8 * width * 2;
        auto number = [](double value) {
            std::ostringstream text;
            text << std::uppercase << std::scientific << std::setprecision(10)
                 << std::setw(17) << value << ' ';
            return text.str();
        };
        for (int record = 0; record < 4; ++record) {
            std::string text;
            for (int slot = 0; slot < slots; ++slot)
                text += number(record * 1000 + slot);
            if (!ocean) {
                for (int row = 0; row < 2; ++row)
                    for (int variable = 6; variable < 8; ++variable)
                        text.replace(
                            18 * (row * 8 * width + variable * width + 4),
                            18 * (width - 4), 18 * (width - 4), ' ');
            }
            if (!record) {
                const std::vector<double> header =
                    ocean ? std::vector<double>{16, 2, .1,   1, 20,
                                                10, 5, 9.81, 1, 1}
                          : std::vector<double>{16, 4, 2, .1, 1, 20, 10, 5, 0};
                for (std::size_t i = 0; i < header.size(); ++i)
                    text.replace(i * 18, 18, number(header[i]));
            } else if (bad_derivative) {
                text.replace(
                    18 * 3 * width * components, 18, "       not-a-mode ");
                // Keep the exact slot width, including padding.
                text.resize(18 * slots, ' ');
            }
            file << text;
        }
    }
};

template <typename VT>
void compare_reads(const std::string& filename, bool ocean)
{
    ReadModes<VT> full(filename, ocean, true), brief(filename, ocean);
    ReadModes<VT> direct;
    ASSERT_TRUE(direct.initialize(filename, ocean, false, false));
    const int n = full.get_vector_size(), na = full.get_addl_vector_size();
    std::vector<VT> x(n), y(n), z(n), t(n), fs(n), fst(n), add(na), addt(na);
    std::vector<VT> bx, by, bz, bfs, ba, dx, dy, dz, dfs, da;
    std::vector<VT> rx, ry, rz, rfs, ra, rt, rfst, radd;
    ReadModes<VT> rates(filename, ocean);
    for (int step : {2, -1, 0, 1, 2, 0}) {
        ASSERT_TRUE(full.get_data(step, x, y, z, t, fs, fst, add, addt));
        ASSERT_TRUE(rates.read_into(
            step, rx, ry, rz, rfs, ocean ? nullptr : &ra, &rt, &rfst,
            ocean ? nullptr : &radd));
        EXPECT_EQ(x, rx);
        EXPECT_EQ(fs, rfs);
        EXPECT_EQ(t, rt);
        EXPECT_EQ(fst, rfst);
        if (!ocean) EXPECT_EQ(addt, radd);
        ASSERT_TRUE(
            brief.read_into(step, bx, by, bz, bfs, ocean ? nullptr : &ba));
        ASSERT_TRUE(
            direct.read_into(step, dx, dy, dz, dfs, ocean ? nullptr : &da));
        EXPECT_EQ(x, bx);
        EXPECT_EQ(y, by);
        EXPECT_EQ(z, bz);
        EXPECT_EQ(fs, bfs);
        EXPECT_EQ(add, ba);
        EXPECT_EQ(x, dx);
        EXPECT_EQ(y, dy);
        EXPECT_EQ(z, dz);
        EXPECT_EQ(fs, dfs);
        EXPECT_EQ(add, da);
        if (step == -1)
            for (int i = 0; i < (ocean ? 5 : 7); ++i) EXPECT_EQ(x[i], VT{});
        if (step >= 0) {
            if constexpr (std::is_same_v<VT, double>)
                EXPECT_EQ(x[0], (step + 1) * 1000);
            else
                EXPECT_EQ(x[0], VT((step + 1) * 1000, (step + 1) * 1000 + 1));
        }
    }
    const auto previous = x;
    EXPECT_FALSE(full.get_data(3, x, y, z, t, fs, fst, add, addt));
    EXPECT_EQ(x, previous);
    EXPECT_FALSE(direct.read_into(3, dx, dy, dz, dfs, ocean ? nullptr : &da));
    EXPECT_EQ(x, dx);
    ASSERT_TRUE(full.get_data(0, x, y, z, t, fs, fst, add, addt));
    const auto reads = full.records_read();
    ASSERT_TRUE(full.get_data(0, x, y, z, t, fs, fst, add, addt));
    EXPECT_EQ(full.records_read(), reads);
    auto copy = full;
    ASSERT_TRUE(copy.get_data(1, x, y, z, t, fs, fst, add, addt));
    ASSERT_TRUE(full.get_data(2, x, y, z, t, fs, fst, add, addt));
    EXPECT_THROW(
        direct.read_into(-2, dx, dy, dz, dfs, ocean ? nullptr : &da),
        std::invalid_argument);
    EXPECT_FALSE(direct.read_into(
        std::numeric_limits<int>::max(), dx, dy, dz, dfs,
        ocean ? nullptr : &da));
}

TEST_F(ModesIO, OceanFullBriefDirectBackwardCacheCopyAndEOF)
{
    write(true);
    compare_reads<std::complex<double>>(filename, true);
}

TEST_F(ModesIO, NwtFullBriefDirectBackwardCacheCopyAndEOF)
{
    write(false);
    compare_reads<double>(filename, false);
}

TEST_F(ModesIO, MetadataOnlyHasNoFileOrModeStorage)
{
    write(false);
    ReadModes<double> root(filename, false), replica;
    replica.initialize(root.metadata(), false);
    EXPECT_EQ(replica.metadata().scalars, root.metadata().scalars);
    EXPECT_EQ(replica.get_vector_size(), root.get_vector_size());
    EXPECT_EQ(replica.time2step(.2), root.time2step(.2));
    std::vector<double> x, y, z, fs, add;
    EXPECT_THROW(replica.read_into(0, x, y, z, fs, &add), std::logic_error);
    ReadModes<double> wrong_type;
    EXPECT_THROW(
        wrong_type.initialize(root.metadata(), true), std::invalid_argument);
    auto bad = root.metadata();
    bad.scalars[0] = 0;
    ReadModes<double> bad_dt;
    EXPECT_THROW(bad_dt.initialize(bad, false), std::invalid_argument);
}

TEST_F(ModesIO, SkipsUnusedDerivativeSlots)
{
    for (bool ocean : {true, false}) {
        write(ocean, true);
        if (ocean) {
            ReadModes<std::complex<double>> brief(filename),
                full(filename, true, true);
            std::vector<std::complex<double>> x, y, z, fs;
            EXPECT_TRUE(brief.read_into(0, x, y, z, fs));
            EXPECT_THROW(full.read_data(0), std::runtime_error);
        } else {
            ReadModes<double> brief(filename, false),
                full(filename, false, true);
            std::vector<double> x, y, z, fs, add;
            EXPECT_TRUE(brief.read_into(0, x, y, z, fs, &add));
            EXPECT_THROW(full.read_data(0), std::runtime_error);
        }
    }
}

TEST_F(ModesIO, RejectsMalformedRetainedFieldsAndDoesNotCacheFailure)
{
    write(true);
    std::fstream file(
        filename, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(18 * 12 * 9 * 2);
    file << "    invalid-value ";
    file.close();
    ReadModes<std::complex<double>> reader(filename);
    std::vector<std::complex<double>> x(18, 42), y(18, 42), z(18, 42),
        fs(18, 42);
    EXPECT_THROW(reader.get_data(0, x, y, z, fs), std::runtime_error);
    EXPECT_EQ(x, std::vector<std::complex<double>>(18, 42));
    EXPECT_THROW(reader.get_data(0, x, y, z, fs), std::runtime_error);
    EXPECT_EQ(reader.records_read(), 0);
}

TEST_F(ModesIO, RejectsTruncationRatherThanLoopingItAsEOF)
{
    write(false);
    std::filesystem::resize_file(filename, 18 * 8 * 16 * 2 + 25);
    ReadModes<double> reader(filename, false);
    EXPECT_THROW(reader.read_data(0), std::runtime_error);
    EXPECT_FALSE(reader.read_data(1));
}

TEST_F(ModesIO, MissingFileInitializationCanBeRetried)
{
    ReadModes<double> reader;
    EXPECT_FALSE(reader.initialize(filename + "-missing", false));
    write(false);
    EXPECT_TRUE(reader.initialize(filename, false));
}

TEST_F(ModesIO, RejectsInvalidHeaderAndIncompatibleDestinations)
{
    write(false);
    ReadModes<double> wrong_flag;
    EXPECT_THROW(wrong_flag.initialize(filename, true), std::invalid_argument);
    ReadModes<double> reader(filename, false);
    std::vector<double> x, y, z, fs;
    EXPECT_THROW(reader.read_into(0, x, y, z, fs), std::invalid_argument);
    std::fstream file(
        filename, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(0);
    file << " 0.0000000000E+00 ";
    file.close();
    ReadModes<double> zero_dimension;
    EXPECT_THROW(
        zero_dimension.initialize(filename, false), std::runtime_error);
}
} // namespace
