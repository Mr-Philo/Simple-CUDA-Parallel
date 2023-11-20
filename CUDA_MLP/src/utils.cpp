#include "utils.h"
#include <ctime>
#include <iostream>
#include <string>

using namespace std;

bool debug = false;

void enable_debug_mode() {
    debug = true;
    print_debug("Debug mode is enabled.", "sys");
}

std::string get_time() {
    time_t timer = std::time(nullptr);
    char time_str[20];
    strftime(time_str, 20, "%Y-%m-%d %H:%M:%S", std::localtime(&timer));
    return time_str;
}

void print(const std::string &msg, const std::string &level, const std::string &component = "sys") {
    std::cout << "[" << get_time() << "][" << level << "][" << component << "]: " << msg << std::endl;
}

void print_info(const std::string &msg, const std::string &component) {
    print(msg, "INFO", component);
}

void print_debug(const std::string &msg, const std::string &component) {
    if (debug) {
        print(msg, "DEBUG", component);
    }
}

void print_warn(const std::string &msg, const std::string &component) {
    print(msg, "WARN", component);
}

void print_error(const std::string &msg, const std::string &component) {
    print(msg, "ERROR", component);
}

void print_fatal(const std::string &msg, const std::string &component) {
    print(msg, "FATAL", component);
}
