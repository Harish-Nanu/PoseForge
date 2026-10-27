#pragma once

#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Minimal "--key value" / "--flag" parser for the CLI.
class Args {
public:
    Args(int argc, char** argv, int first, std::set<std::string> flags)
        : flags_(std::move(flags)) {
        for (int i = first; i < argc; ++i) {
            std::string token = argv[i];
            if (token.rfind("--", 0) != 0) {
                throw std::invalid_argument("unexpected argument: " + token);
            }
            const std::string key = token.substr(2);
            if (flags_.contains(key)) {
                values_[key] = "1";
            } else if (i + 1 < argc) {
                values_[key] = argv[++i];
            } else {
                throw std::invalid_argument("missing value for --" + key);
            }
        }
    }

    bool has(const std::string& key) const { return values_.contains(key); }
    bool flag(const std::string& key) const { return has(key); }

    std::string str(const std::string& key, const std::string& fallback) const {
        const auto it = values_.find(key);
        return it == values_.end() ? fallback : it->second;
    }
    int integer(const std::string& key, int fallback) const {
        return has(key) ? std::stoi(values_.at(key)) : fallback;
    }
    double number(const std::string& key, double fallback) const {
        return has(key) ? std::stod(values_.at(key)) : fallback;
    }
    std::vector<std::size_t> sizes(const std::string& key, std::vector<std::size_t> fallback) const {
        if (!has(key)) return fallback;
        std::vector<std::size_t> out;
        std::string item;
        for (const char c : values_.at(key) + ",") {
            if (c == ',') {
                if (!item.empty()) out.push_back(static_cast<std::size_t>(std::stoul(item)));
                item.clear();
            } else {
                item += c;
            }
        }
        return out;
    }

    // Rejects options the command doesn't know, so typos fail loudly.
    void allow_only(const std::set<std::string>& known) const {
        for (const auto& [key, value] : values_) {
            if (!known.contains(key) && !flags_.contains(key)) {
                throw std::invalid_argument("unknown option --" + key);
            }
        }
    }

private:
    std::set<std::string> flags_;
    std::map<std::string, std::string> values_;
};
