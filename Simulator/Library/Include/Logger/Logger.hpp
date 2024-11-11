/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_LOGGER_HPP
#define VT_PHYSICS_LOGGER_HPP

#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

#ifdef NDEBUG

#define LOG_INFO(msg) {};

#define LOG_WARNING(msg) {};

#define LOG_ERROR(msg) {};

#else

#endif

#define LOG_INFO(msg) \
    spdlog::info(msg);

#define LOG_WARNING(msg) \
    spdlog::warn(msg);

#define LOG_ERROR(msg) \
    spdlog::error(msg);

#endif //VT_PHYSICS_LOGGER_HPP
