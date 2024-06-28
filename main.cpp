#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <semaphore>
#include <fstream>

using namespace std;
using namespace cv;

const int filterSize = 3;
const int chunkRows = 4;
const int chunkCols = 4;

counting_semaphore<1> semaphores[16] = {
		counting_semaphore<1>(1), counting_semaphore<1>(0), counting_semaphore<1>(0), counting_semaphore<1>(0),
		counting_semaphore<1>(0), counting_semaphore<1>(-1), counting_semaphore<1>(-1), counting_semaphore<1>(-1),
		counting_semaphore<1>(0), counting_semaphore<1>(-1), counting_semaphore<1>(-1), counting_semaphore<1>(-1),
		counting_semaphore<1>(0), counting_semaphore<1>(-1), counting_semaphore<1>(-1),  counting_semaphore<1>(-1)
};

Mat GridToImage(int* grid, int rows, int cols);
void writeGridToFile(const int* grid, int rows, int cols, const std::string& filename);
void showImage(Mat image);

int calculateFilter(int row, int col, int* grid, int cols, const int filters[][filterSize][filterSize], int filterNum){
	long long filterValues[filterNum];
	long long ans = 0;

	for(int f=0; f<filterNum; f++){
		filterValues[f] = 0;
		for(int i = 0; i < filterSize; ++i){
			for(int j = 0; j < filterSize; ++j){
				filterValues[f] += grid[(row + i) * cols + (col + j)] * filters[f][i][j];
			}
		}
		ans += (filterValues[f] * filterValues[f]);
	}
	return round(sqrt(ans));
}

void process(int srow, int drow, int scol, int dcol, int* grid, int gridcol, 
			const int filters[][filterSize][filterSize], int filterNum, int id, int bottom, int right){
	
	int rows = drow - srow + 1;
	int cols = dcol - scol + 1;

	int rowbuffer[2][cols];
	int colbuffer[2][rows-2];

	// writing to buffer
	for(int j=scol; j<=dcol; j++){
		for(int i=drow; i>=drow-1; i--){
			rowbuffer[i-drow+1][j-scol] = calculateFilter(i, j, grid, gridcol, filters, filterNum);
		}
	}
	if(bottom != -1){
		semaphores[bottom].release();
	}


	for(int i=srow; i<=drow-2; i++){
		for(int j=dcol; j>=dcol-1; j--){
			colbuffer[j-dcol+1][i-srow] = calculateFilter(i, j, grid, gridcol, filters, filterNum);
		}
	}
	if(right != -1){
		semaphores[right].release();
	}

	// aquire semaphore
	semaphores[id].acquire();

	for(int i=srow; i<=drow-2; i++){
		for(int j=scol; j<=dcol-2; j++){
			grid[i * gridcol + j] = calculateFilter(i, j, grid, gridcol, filters, filterNum);
		}
	}

	// writing back from the buffer
	for(int j=scol; j<=dcol; j++){
		for(int i=drow; i>=drow-1; i--){
			grid[i * gridcol + j] = rowbuffer[i-drow+1][j-scol];
		}
	}
	for(int i=srow; i<=drow-2; i++){
		for(int j=dcol; j>=dcol-1; j--){
			grid[i * gridcol + j] = colbuffer[j-dcol+1][i-srow];
		}
	}
}

int main() {
	string s;
	cin >> s;
	Mat image = imread(s, IMREAD_COLOR);

	if (image.empty()) {
		std::cout << "Could not open or find the image" << endl;
		return -1;
	}

	int rows = image.rows;
	int cols = image.cols;
	int channels = image.channels();
	std::cout << "rows & cols of image: " << rows << " " << cols << endl;

	if(channels < 3){
		std::cout << "Not an RGB photo!" << endl;
		return -1;
	}

	int* grid = new int[rows * cols];

	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int value = 0.2989*image.at<Vec3b>(r, c)[0] + 0.5870*image.at<Vec3b>(r, c)[1] + 0.1140*image.at<Vec3b>(r, c)[2];
			grid[r * cols + c] = value;
		}
	}

	const int filters[][filterSize][filterSize] = {
		{{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}},
		{{-1, -2, -1},{0, 0, 0},{1, 2, 1}},
		{{0, 1, 2},{-1, 0, 1},(-2, -1, 0)},
		{{-2, -1, 0},{-1, 0, 1},{0, 1, 2}},
	};

	int chunkRowSize = (rows - filterSize + 1) / chunkRows;
	int chunkColSize = (cols - filterSize + 1) / chunkCols;

	thread threads[chunkRows][chunkCols];

	for(int i=0; i<chunkRows; i++){
		for(int j=0; j<chunkCols; j++){
			int srow = i * chunkRowSize;
			int drow = (i+1) * chunkRowSize - 1;
			int scol = j * chunkColSize;
			int dcol = (j+1) * chunkColSize - 1;
			if(i == chunkRows - 1) drow = rows-filterSize;
			if(j == chunkCols - 1) dcol = cols-filterSize;
			
			int id = i * chunkCols + j;
			int bottom = (i+1) * chunkCols + j;
			int right = i * chunkCols + j + 1;
			if(i == chunkRows-1) bottom = -1;
			if(j == chunkCols-1) right = -1;

			threads[i][j] = thread(process, srow, drow, scol, dcol, grid, cols, filters, 4, id, bottom, right);
		}
	}

	for(int i=0; i<chunkRows; i++){
		for(int j=0; j<chunkCols; j++){
			threads[i][j].join();
		}
	}

	long long sum = 0;
	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			sum += grid[i * cols + j];	
		}
	}
	sum = round( (double)sum / (rows * cols));

	for(int i=0; i<rows; i++){
		for(int j=0; j<cols; j++){
			grid[i * cols + j] = (grid[i * cols + j] >= sum)*255;
		}
	}

	imwrite("created_image.jpg", GridToImage (grid, rows, cols));
	showImage(GridToImage(grid, rows, cols));

	delete[] grid;
	return 0;
}


Mat GridToImage(int* grid, int rows, int cols){
	Mat image(rows, cols, CV_8UC1);
	// std::cout << "rows & cols of the outputimages image: " << rows << " " << cols << endl;

	// Copy the values from the grid to the Mat
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			image.at<uint8_t>(r, c) = grid[r * cols + c];
		}
	}

	return image;
}

void writeGridToFile(const int* grid, int rows, int cols, const std::string& filename) {
    ofstream outFile(filename);
    if (!outFile) {
        cerr << "Failed to open file for writing" << endl;
        return;
    }

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            outFile << grid[r * cols + c] << "\t";
        }
        outFile << endl;
    }

    outFile.close();
}

void showImage(Mat image){
	imshow("Grid Image", image);
	waitKey(0); // Wait for a key press to close the window
}
