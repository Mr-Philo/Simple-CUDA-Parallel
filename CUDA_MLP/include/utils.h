#ifndef CPP_TEMPLATE_UTILS_H
#define CPP_TEMPLATE_UTILS_H

#include <string>
#include <ctime>

std::string get_time();

void enable_debug_mode();

void print(const std::string &msg, const std::string &component);
void print_info(const std::string &msg, const std::string &component);
void print_debug(const std::string &msg, const std::string &component);
void print_warn(const std::string &msg, const std::string &component);
void print_error(const std::string &msg, const std::string &component);
void print_fatal(const std::string &msg, const std::string &component);

#endif //CPP_TEMPLATE_UTILS_H
