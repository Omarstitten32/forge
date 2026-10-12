#include <catch2/catch_test_macros.hpp>
#include "forge/core/error.hpp"
#include "forge/core/log.hpp"

TEST_CASE("Logging initializes cleanly", "[core]") {
    REQUIRE_NOTHROW(forge::init_logging());
}

TEST_CASE("ForgeError carries message", "[core]") {
    forge::ForgeError err("hello");
    REQUIRE(std::string(err.what()) == "hello");
}

TEST_CASE("ParseError prefixes message", "[core]") {
    forge::ParseError err("bad input");
    std::string msg = err.what();
    REQUIRE(msg.find("ParseError") != std::string::npos);
    REQUIRE(msg.find("bad input")  != std::string::npos);
}
