//
// Created by artemisia on 6/15/23.
//

#include <dlfcn.h>
#include <filesystem>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <unistd.h>

#define _GNU_SOURCE

namespace fs = std::filesystem;

using std::string, std::stringstream;

unsigned long hash(const string &name) {
    unsigned long hash = 5381;
    for (char c: name) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

int open(const char *pathname, int flags) {
    typedef int (*real_open)(const char *pathname, int flags);
    auto actual_open = (real_open) dlsym(RTLD_NEXT, "open");
    string str(pathname);
    std::string base_filename = str.substr(str.find_last_of("/\\") + 1);
    std::string dir_name = str.substr(0, str.find_last_of("/\\"));
    if (fs::exists("/data/ml/" + base_filename) || dir_name!="/data/ml") {
        return actual_open(str.c_str(), flags);
    } else {
        std::unordered_map<unsigned long, string> hash_to_ip;
        auto ipfile = popen("LD_PRELOAD='' cat test.txt", "r");
        string file, row;
        char buf[16];
        while (fgets(buf, 16, ipfile)) {
            file += buf;
        }
        pclose(ipfile);

        stringstream row_reader(file);
        while (getline(row_reader, row)) {
            auto ip = row.substr(row.find(','));
            unsigned long rank = std::stol(row.substr(0, row.find(',')));
            hash_to_ip[rank] = ip;
        }

        auto hsh = hash(base_filename);
        auto size = hash_to_ip.size();
        string prep="LD_PRELOAD='' split -n 3 -a 2 -d "+base_filename+" "+base_filename;
        string base_cmd = "LD_PRELOAD='' wget -P /dev/shm http://" + hash_to_ip[(hsh % size)] + base_filename;
//        system((base_cmd+"0").c_str());
//        system((base_cmd+"1").c_str());
//        system((base_cmd+"2").c_str());
        str = string("/dev/shm/" + base_filename);
        return actual_open(str.c_str(), flags);
    }
}

int close(int fd) {
    typedef int (*real_close)(int fd);
    auto actual_close = (real_close) dlsym(RTLD_NEXT, "close");
    string fd_path("/proc/self/");
    fd_path += std::to_string(fd);
    char file_path[100];
    if (readlink(fd_path.c_str(), file_path, 99) == -1) {
        return -1;
    }
    int out = actual_close(fd);
    string fp(file_path);
    string dir = fp.substr(0, fp.find_last_of("/\\"));
    if (dir == "/dev/shm") {
        string cmd = "LD_PRELOAD='' rm " + fp;
        system(cmd.c_str());
    }
    return out;
}
