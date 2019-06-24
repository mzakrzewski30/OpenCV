#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <limits>
#define M_PI 3.14159265358979323846

using namespace cv;
using namespace std;

string path = "C:\\CVLab\\Project\\Project\\";
string file = "5.bmp";
string contour = "rectangle";
double mu = 1.0;
double nu = 65.025;
double lambda1 = 1.0;
double lambda2 = 1.0;
double epsilon = 1.0;
double timestep = 0.05;
double iterNum = 1000;
double sigma = 3.0;
int RSF = 1;
int LRCV = 1;
int LIF = 1;
int isExchange = 0;

int save_img_final = 1;
int save_img_all = 1;
int save_contour_final = 1;
int save_contour_all = 1;
int read_contour = 0;


Mat get_initial_contour(Size size, string file) {
	ifstream in(file, ifstream::in);

	string contour_type;
	in >> contour_type;

	Mat_<double> initial_lsf = Mat::ones(size, CV_64FC1);

	if (contour_type == "rectangle") {
		Point top_left, bot_right;
		in >> top_left.x >> top_left.y >> bot_right.x >> bot_right.y;
		initial_lsf(Rect(top_left, bot_right)).setTo(-1.0);
	}
	else if (contour_type == "ellipse") {
		Point center;
		Size size;
		in >> center.x >> center.y;
		in >> size.height >> size.width;

		ellipse(initial_lsf, center, size, 0, 0, 360, Scalar(-1.0), CV_FILLED);
	}
	else if (contour_type == "circle") {
		Point center;
		int radius;
		in >> center.x >> center.y;
		in >> radius;
		circle(initial_lsf, center, radius, Scalar(-1.0), CV_FILLED);
	}
	else if (contour_type == "contour") {
		vector<Point> contour;
		int x, y;
		while (in >> x >> y) {
			Point point(x, y);
			contour.push_back(point);
		}

		vector<vector<Point> > contours;
		contours.push_back(contour);

		drawContours(initial_lsf, contours, -1, Scalar(-1.0), CV_FILLED);
		in.close();
	}
	else {
		cout << "wrong contour type" << endl;
		exit(1);
	}

	return initial_lsf;
}

void write_contours_to_file(vector<vector<Point> > contours, string filename) {
	ofstream fout(filename);
	int i;
	for (i = 0; i < contours.size(); i++) {
		vector<Point> contour = contours[i];
		fout << "contour " << i + 1 << endl;

		for (Point p : contour) {
			fout << p.x << " " << p.y << endl;
		}
	}
}

Mat read_contours_from_file(Mat Img, string filename,string path) {
	string x;
	int y;
	vector<Point> contour;
	vector<vector<Point> > contours;
	ifstream in(filename, ifstream::in);
	in >> x >> y;
	while (in >> x >> y) {
		if (x == "contour") {
			contours.push_back(contour);
			contour.clear();
		}
		else {
			Point point(stoi(x), y);
			contour.push_back(point);
		}
	}
	contours.push_back(contour);
	Mat img;
	Img.copyTo(img);
	cvtColor(Img, Img, CV_GRAY2BGR);
	drawContours(Img, contours, -1, Scalar(0, 0, 255));
	resize(Img, Img, Size(400, 400));
	imshow("read", Img);
	string save_path = path + "contour_form_file" + ".jpg";
	imwrite(save_path, Img);
	Mat_<double> initial_lsf = Mat::ones(img.size(), CV_64FC1);
	drawContours(initial_lsf, contours, -1, Scalar(-1.0), CV_FILLED);
	return initial_lsf;
}

Mat neumann_boundary_condition(const Mat &in) {
	Mat mat = in.clone();
	// corners
	mat.at<double>(Point(0, 0)) = mat.at<double>(Point(2, 2));
	mat.at<double>(Point(mat.cols - 1, 0)) = mat.at<double>(Point(mat.cols - 3, 2));
	mat.at<double>(Point(0, mat.rows - 1)) = mat.at<double>(Point(2, mat.rows - 3));
	mat.at<double>(Point(mat.cols - 1, mat.rows - 1)) = mat.at<double>(Point(mat.cols - 3, mat.rows - 3));

	// edges
	Rect top_edge(Point(1, 0), Size(mat.cols - 2, 1));
	Rect bottom_edge(Point(1, mat.rows - 1), Size(mat.cols - 2, 1));
	Rect one_of_top_rows(Point(1, 2), Size(mat.cols - 2, 1));
	Rect one_of_bottom_rows(Point(1, mat.rows - 3), Size(mat.cols - 2, 1));
	mat(one_of_top_rows).copyTo(mat(top_edge));
	mat(one_of_bottom_rows).copyTo(mat(bottom_edge));
	Rect left_edge(Point(0, 1), Size(1, mat.rows - 2));
	Rect right_edge(Point(mat.cols - 1, 1), Size(1, mat.rows - 2));
	Rect one_of_left_cols(Point(2, 1), Size(1, mat.rows - 2));
	Rect one_of_right_cols(Point(mat.cols - 3, 1), Size(1, mat.rows - 2));

	mat(one_of_left_cols).copyTo(mat(left_edge));
	mat(one_of_right_cols).copyTo(mat(right_edge));

	return mat;
}

Mat curvature_central(Mat u) {
	Mat K(u.size(), u.type());

	Mat ux, uy;
	Sobel(u, ux, -1, 1, 0, 1, 0.5);
	Sobel(u, uy, -1, 0, 1, 1, 0.5);

	Mat normDu = (ux.mul(ux) + uy.mul(uy)) + 1e-10;
	sqrt(normDu, normDu);

	Mat Nx = ux.mul(1 / normDu);
	Mat Ny = uy.mul(1 / normDu);

	Mat nxx, nyy;

	Sobel(Nx, nxx, -1, 1, 0, 1, 0.5);
	Sobel(Ny, nyy, -1, 0, 1, 1, 0.5);

	K = nxx + nyy;

	return K;
}

pair<Mat, Mat> exchange(Mat f1, Mat f2, int isExchange) {
	Mat f1_min(f1.size(), CV_64FC1), f2_max(f2.size(), CV_64FC1);
	if (isExchange == 0) {
	}
	for (int i = 0; i < f1.rows; i++) {
		for (int j = 0; j < f1.cols; j++) {
			double f1_val = f1.at<double>(Point(j, i));
			double f2_val = f2.at<double>(Point(j, i));
			double elem = f1_val;
			if (isExchange == 1) {
				if (f2_val < f1_val) {
					elem = f2_val;
				}
			}
			else if (isExchange == -1) {
				if (f2_val > f1_val) {
					elem = f2_val;
				}
			}
			f1_min.at<double>(Point(j, i)) = elem;
		}
	}
	for (int i = 0; i < f2.rows; i++) {
		for (int j = 0; j < f2.cols; j++) {
			double f1_val = f1.at<double>(Point(j, i));
			double f2_val = f2.at<double>(Point(j, i));
			double elem = f2_val;
			if (isExchange == 1) {
				if (f1_val > f2_val) {
					elem = f1_val;
				}
			}
			else if (isExchange == -1) {
				if (f1_val < f2_val) {
					elem = f1_val;
				}
			}
			f2_max.at<double>(Point(j, i)) = elem;
		}
	}
	return make_pair(f1_min, f2_max);
}

tuple<Mat_<double>, Mat_<double>, Mat_<double>> ACM(Mat_<double> u, Mat_<double> Img,
	Mat_<double> Ksigma, Mat_<double> KI, Mat_<double> KI2, Mat_<double> KONE,
	double nu, double timestep, double mu, double epsilon, double lambda1,
	double lambda2, int RSF, int LRCV, int LIF, int isExchange, Mat_<double> log_) {

	u = neumann_boundary_condition(u);
	Mat_<double> K = curvature_central(u);

	Mat Hu(u.size(), CV_64FC1, Scalar::all(0));
	for (int i = 0; i < Hu.rows; i++) {
		for (int j = 0; j < Hu.cols; j++) {
			double u_elem = u.at<double>(Point(j, i));
			Hu.at<double>(Point(j, i)) = 0.5 * (1 + (2 / M_PI) * atan(u_elem / epsilon));
		}
	}
	Mat_<double> DrcU = 1 / (u.mul(u) + epsilon * epsilon) * epsilon / M_PI;

	Mat_<double> KIH;
	filter2D(Hu.mul(Img), KIH, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);

	Mat_<double> KH;
	filter2D(Hu, KH, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);

	Mat_<double> f1 = KIH / KH;
	Mat_<double> f2 = (KI - KIH);
	divide(f2, KONE - KH, f2);

	auto ret = exchange(f1, f2, isExchange);
	f1 = ret.first;
	f2 = ret.second;

	Mat_<double> LRCVterm;
	if (LRCV != 0) {
		Mat_<double> tempf1 = Img - f1;
		Mat_<double> tempf2 = Img - f2;
		LRCVterm = LRCV * DrcU.mul((-lambda1 * tempf1.mul(tempf1) + lambda2 * tempf2.mul(tempf2)));
	}
	else
		LRCVterm = 0;

	Mat_<double> LIFterm;
	if (LIF != 0)
		LIFterm = DrcU.mul(((Img - f1.mul(Hu) - f2.mul((1 - Hu))).mul((f1 - f2))));
	else
		LIFterm = 0;

	Mat_<double> RSFterm;
	if (RSF != 0) {
		Mat_<double> s1 = lambda1 * f1.mul(f1) - f2.mul(f2);
		Mat_<double> s2 = lambda1 * f1 - lambda2 * f2;
		Mat_<double> dataForce, filtered_s1, filtered_s2;
		filter2D(s1, filtered_s1, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);
		filter2D(s2, filtered_s2, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);
		dataForce = (lambda1 - lambda2) * KONE.mul(Img).mul(Img) + filtered_s1 - 2 * Img.mul(filtered_s2);
		RSFterm = -RSF * DrcU.mul(dataForce);
	}
	else {
		RSFterm = 0;
	}

	Mat_<double> LoGterm = log_.mul(DrcU);

	Mat_<double> laplacian_u;
	Laplacian(u, laplacian_u, -1, 1, 0.25);
	laplacian_u.convertTo(laplacian_u, CV_64FC1);

	Mat_<double> PenaltyTerm = (4 * laplacian_u - K) * mu;
	Mat_<double> LengthTerm = nu * DrcU.mul(K);
	u = u + timestep * (LengthTerm + PenaltyTerm + RSFterm + LRCVterm + LIFterm + LoGterm);

	return make_tuple(u, f1, f2);
}

void prog(){
	Mat Img = imread(path+"img\\"+file, IMREAD_GRAYSCALE);
	cout << Img.size() << endl;
	Mat_<double> initial_lsf;
	if (read_contour == 1) {
		initial_lsf = read_contours_from_file(Img, path + "contours.txt", path);
	}
	else {
		initial_lsf = get_initial_contour(Img.size(), path + "contour\\" + contour);
	}


	Img = imread(path + "img\\" + file, IMREAD_GRAYSCALE);

	Mat_<double> u = initial_lsf;

	Mat_<double> gauss_kernel_1d = getGaussianKernel(round(2 * sigma) * 2 + 1, sigma, CV_64FC1);
	Mat_<double> gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t();
	Mat_<double> Ksigma = gauss_kernel_2d;

	Mat_<double> KONE;
	filter2D((Mat::ones(Img.size(), CV_64FC1)), KONE, CV_64FC1, Ksigma);

	Mat_<double> KI;
	filter2D(Img, KI, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);

	Mat_<double> KI2;
	filter2D(Img.mul(Img), KI2, CV_64FC1, Ksigma, Point(-1, -1), 0, BORDER_REPLICATE);

	
	Mat_<double> G = getGaussianKernel(9, 1, CV_64FC1);
	Mat_<double> Img_gao;
	filter2D(Img, Img_gao, CV_64FC1, G, Point(-1, -1), 0, BORDER_REPLICATE);
	Mat Ix, Iy;
	Sobel(Img_gao, Ix, -1, 1, 0, 1, 0.5);
	Sobel(Img_gao, Iy, -1, 0, 1, 1, 0.5);
	Mat_<double> f = Ix.mul(Ix) + Iy.mul(Iy);
	Mat_<double> laplacian_Img;
	Laplacian(Img_gao, laplacian_Img, -1, 1, 0.25);
	laplacian_Img.convertTo(laplacian_Img, CV_64FC1);
	Mat_<double> log = 4 * laplacian_Img;
	Mat_<double> g_ = Mat::zeros(Img.size(), CV_64FC1);
	Mat_<double> log_ = Mat::zeros(Img.size(), CV_64FC1);
	Mat_<double> g;
	exp(-0.01*f, g);
	for (int i = 1; i < 100; i++) {
		log_ = log_ + 0.01*(g.mul(log_) - (1 - g).mul(log_ - 5 * log));
	}
	log_ = 20 * log_;
	

	Mat_<double> f1, f2, img_copy;
	vector<vector<Point> > final_contours;
	Mat final_image;
	Img.convertTo(img_copy, CV_64FC1);

	for (int n = 0; n < iterNum; n++) {
		auto acm_result = ACM(u, img_copy, Ksigma, KI, KI2, KONE, nu, timestep, mu, epsilon,
			lambda1, lambda2, RSF, LRCV, LIF, isExchange,log_);
		u = get<0>(acm_result);
		f1 = get<1>(acm_result);
		f2 = get<2>(acm_result);

		if (n % 1 == 0) {
			Mat new_image = Img.clone();
			Mat contour_mat;

			double limit = 0.0;
			threshold(u, contour_mat, limit, 1, THRESH_BINARY);
			//imshow("debug", contour_mat);
			contour_mat.convertTo(contour_mat, CV_8UC1, 255.0);

			// swap negative/positive
			contour_mat = contour_mat * -1 + 255;

			vector<vector<Point> > contours;
			vector<vector<Point> > large_contours;

			findContours(contour_mat, contours, RETR_LIST, CHAIN_APPROX_NONE);
			if (contours.size() != 0) {
				for (int i = 0; i < contours.size(); i++) {
					double a = contourArea(contours[i], false);
					if (a >= 100) {
						large_contours.push_back(contours[i]);
					}
				}
			}
			contours = large_contours;
			cvtColor(new_image, new_image, CV_GRAY2BGR);
			drawContours(new_image, contours, -1, Scalar(0, 0, 255));
			resize(new_image, new_image, Size(400, 400));
			putText(new_image, "Iter: " + to_string(n), Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

			if (save_img_all) {
				string iter_num;
				iter_num += std::to_string(n);
				string filename = path + "recImg\\" + "iter_" + iter_num + ".jpg";
				imwrite(filename, new_image);
			}

			if (save_contour_all) {
				string iter_num;
				iter_num += std::to_string(n);
				string final_contours_filename = path + "recContour\\" + "iter_" + iter_num + ".txt";
				if (final_contours.size() > 0)
					write_contours_to_file(final_contours, final_contours_filename);
			}

			if (save_contour_final)
				final_contours = contours;

			if (save_img_final)
				final_image = new_image;

			imshow("window", new_image);

			int wait_time = 200;
			if (waitKey(wait_time) == 27)
				break;
		}
	}

	if (save_contour_final) {
		string final_contours_filename = path + "recContour\\" + "final_contour" + ".txt";
		if (final_contours.size() > 0)
			write_contours_to_file(final_contours, final_contours_filename);
	}


	string filename = path + "recImg\\" + "final_Img" + ".jpg";
	imwrite(filename, final_image);

	while (true)
		if (waitKey(1) == 27)
			break;
	cv::destroyAllWindows();
}


int main() {
	int choice;
	int mode;
	int stop = 0;
	string file_num;
	while (stop == 0) {
		system("CLS");
		std::cout << "0) Uruchom" << endl;
		std::cout << "1) Wybierz plik" << endl;
		std::cout << "2) Wybierz kontur startowy" << endl;
		std::cout << "3) Zapis obrazu" << endl;
		std::cout << "4) Zapis konturu" << endl;
		std::cout << "5) Wyjscie z programu" << endl;

		std::cin >> choice;
		switch (choice) {
		case 0:
			prog();
			break;
		case 1:
			std::cout << "Podaj numer pliku" << endl;

			std::cout << "Domyslnie 1-6" << endl;
			std::cin >> mode;
			file_num = std::to_string(mode);
			file = file_num + ".bmp";
			break;
		case 2:
			std::cout << "1)Podstawowy" << endl;
			std::cout << "2)Wygenerowany" << endl;
			std::cin >> mode;
			if (mode==1) {
				read_contour = 0;
				std::cout << "rectangle" << endl;
				std::cout << "ellipse" << endl;
				std::cout << "contour" << endl;
				std::cout << "circle" << endl;
				std::cin >> contour;
			}
			else {
				read_contour = 1;
			}
			break;
		case 3:
			std::cout << "1) Zapisz ostatni" << endl;
			std::cout << "2) Zapisz wszystkie" << endl;
			std::cin >> mode;
			if (mode==1) {
				save_img_final = 1;
				save_img_all = 0;
			}
			else {
				save_img_final = 0;
				save_img_all = 1;
			}
			break;
		case 4:
			std::cout << "1) Zapisz ostatni" << endl;
			std::cout << "2) Zapisz wszystkie" << endl;
			std::cin >> mode;
			if (mode==1) {
				save_contour_final = 1;
				save_contour_all = 0;
			}
			else {
				save_contour_final = 0;
				save_contour_all = 1;
			}
			break;
		case 5:
			stop = 1;
			break;
		default:
			break;
		}
	}
}