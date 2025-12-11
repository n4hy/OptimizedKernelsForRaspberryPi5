#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <Eigen/Dense>
#include "optmath/neon_kernels.hpp"

// Simple test runner
#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        std::cerr << "Assertion failed: " << #a << " != " << #b << " (" << (a) << " != " << (b) << ")\n"; \
        return 1; \
    }

#define ASSERT_NEAR(a, b, eps) \
    if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "Assertion failed: " << #a << " approx " << #b << "\n"; \
        return 1; \
    }

int main() {
    // Basic NEON add test (wrappers fall back to CPU if NEON disabled)
    Eigen::VectorXf a(3); a << 1.0f, 2.0f, 3.0f;
    Eigen::VectorXf b(3); b << 4.0f, 5.0f, 6.0f;

    Eigen::VectorXf c = optmath::neon::neon_add(a, b);

    ASSERT_EQ(c.size(), 3);
    ASSERT_NEAR(c[0], 5.0f, 1e-5f);
    ASSERT_NEAR(c[1], 7.0f, 1e-5f);
    ASSERT_NEAR(c[2], 9.0f, 1e-5f);

    std::cout << "Tests passed.\n";
    return 0;
}
