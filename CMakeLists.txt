cmake_minimum_required(VERSION 3.0.0)
project(PlateChars VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
find_package(Tesseract 4)
if (NOT Tesseract_FOUND)
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(Tesseract REQUIRED IMPORTED_TARGET tesseract lept)
    add_library(Tesseract::libtesseract ALIAS PkgConfig::Tesseract)
endif ()

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(PlateChars main.cpp)
target_link_libraries( PlateChars ${OpenCV_LIBS} Tesseract::libtesseract)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
