#include <cstdlib>
#include <string_view>

#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>

int main(int argc, char** argv) {
    // OpenCV's internal thread pool (TBB) is not built with ThreadSanitizer, so TSan reports
    // false races inside it. The TSan CI job sets this to test only PoseForge's own threading.
    if (const char* serial = std::getenv("POSEFORGE_SERIAL_OPENCV");
        serial != nullptr && std::string_view(serial) == "1") {
        cv::setNumThreads(0);
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
