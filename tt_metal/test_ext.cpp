#include "tt_metal.cpp"

using namespace tt;
using namespace tt::tt_metal;


int main(void) {

    IDevice* device = CreateDevice(0);

    CloseDevice(device);
}