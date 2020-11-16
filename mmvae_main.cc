#include "mmvae.hh"

int
main(const int argc, const char *argv[])
{
    torch::Tensor tensor = torch::rand({ 2, 3 });
    std::cout << tensor << std::endl;


    ogzstream ofs("temp.txt.gz", std::ios::out);
    ofs.close();


    TLOG("DONE");
    return EXIT_SUCCESS;
}
