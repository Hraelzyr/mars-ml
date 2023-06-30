//
// Created by artemisia on 6/14/23.
//

#include <unordered_map>
#include <string>
#include <fstream>
#include <filesystem>

unsigned long hash(const std::string &name) {
    unsigned long hash = 5381;
    for (char c: name) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

unsigned int self_rank(std::unordered_map<int, std::string>& ip_to_rank){
    auto pptr = popen("ip -4 addr show ib0 | grep -oP "
                      "'(?<=inet\\s)\\d+(\\.\\d+){3}'", "r");
    char ip[16];
    fgets(ip, 15, pptr);
    pclose(pptr);
    auto IP = std::string(ip);
    for (const auto& kv: ip_to_rank){
        if (IP==kv.second) return kv.first;
    }
}

using std::string, std::stringstream;
namespace fs = std::filesystem;

int main() {

    std::unordered_map<int, string> hash_to_ip;
    auto ipfile = std::ifstream("test.txt");

    string row;
    while (getline(ipfile, row)){
        auto buf = stringstream(row);
        string rank_str, ip;
        getline(buf, rank_str, ',');
        getline(buf, ip);
        int rank = std::stoi(rank_str);
        hash_to_ip[rank]=ip;
    }
    ipfile.close();

    unsigned long size = hash_to_ip.size();
    auto src_enum = fs::directory_iterator("../data");
    auto dest = fs::path("/data/ml");

    auto r = self_rank(hash_to_ip);
    for (const auto& file: src_enum){
        if (hash(file.path().filename())%size==r){
            fs::copy(file.path(), dest);
        }
    }

    return 0;
}