#include "fsrcnn.h"
#include <time.h>

int main(int, char**)
{
    FSRCNN* fsrcnn = FSRCNN::getInstance();
    Mat image = imread("C:\\Users\\BONITO\\source\\repos\\FSRCNN-CUDA\\image\\im_2.bmp");
    fsrcnn->init(image.rows, image.cols);
    clock_t start = clock();
    image = fsrcnn->sr(image);
    cout << "process time = " << clock() - start << endl;
    imwrite("C:\\Users\\BONITO\\source\\repos\\FSRCNN-CUDA\\image\\im_2_SR.bmp", image);
}