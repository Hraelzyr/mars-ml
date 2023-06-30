#include <iostream>
#include <mpi.h>
using namespace std;
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    auto pptr = popen("ip -4 addr show ib0 | grep -oP "
                      "'(?<=inet\\s)\\d+(\\.\\d+){3}'", "r");
    char ip[16];
    fgets(ip, 15, pptr);
    pclose(pptr);
    cout<<rank<<","<<ip;
    MPI_Finalize();
    return 0;
}
