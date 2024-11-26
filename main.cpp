#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <xlnt/xlnt.hpp>


int plot(std::vector<double> x, std::vector<double> y, const std::string& title, const std::string& xlabel, const std::string& ylabel);

double activate_func(double x,double a) {
    return 1 / (1 + exp(-x * a));
}

double delta(double x, double a) {
    return a * x * (1 - x);
}

double find_sum(std::vector<double>& weight, std::vector<double>& input) {
    double sum = 0;
    for (int j = 0; j < weight.size(); ++j) {
        sum += weight[j] * input[j];
    }
    return sum;
}

double out_n(std::vector<double> input_w,std::vector<double> input,double a) {
    return activate_func(find_sum(input_w, input),a);
}

int learn(std::vector<double>& weight, std::vector<std::vector<double>>& input_matrix, double learn_rate, double err_exit, double a) {
    std::vector<double> x;
    std::vector<double> y;
    for (int epoch = 0; ; ++epoch) {
        double total_error = 0;
        bool check = true;

        for (int row = 0; row < input_matrix.size(); ++row) {

            std::vector<double> input(input_matrix[row].begin(), input_matrix[row].end() - 1);
            double target = input_matrix[row].back();


            double result = out_n(weight, input, a);

            double err = pow((target - result), 2);
            total_error += err;

            for (size_t j = 0; j < weight.size(); ++j) {
                weight[j] -= learn_rate * (result - target) * input[j] * delta(result, a);
            }
            
           
               

        }
        if (epoch % 100 == 0) std::cout << "epoch: " << epoch << " "  << total_error / input_matrix.size() << std::endl;
        
        total_error /= input_matrix.size();
        x.push_back(epoch);
        y.push_back(total_error);
        if (fabs(total_error) < err_exit) {
            std::cout << total_error << std::endl;
            //plot(x, y, "Error trend during training over generations", "Generation number", "Error");
            return epoch;
        }
    }
}


std::vector<std::vector<double>> read_excel(std::string path) {
    std::vector<std::vector<double>> matrix;
    using namespace xlnt;
    workbook book;
    book.load(path);

    auto w_book = book.active_sheet();
    std::clog << "File opened\n";
    int j = 0;
    for (auto row : w_book.rows(false)) {
        matrix.push_back(std::vector<double>());
        int i = 0;
        for (auto cell : row) {
            if (i == 2) { 
                matrix[j].push_back(1); 
            }
            matrix[j].push_back(std::stod(cell.to_string()));
            ++i;
        }
        ++j;
        
    };
    return matrix;
}




int plot(std::vector<double> x, std::vector<double> y, const std::string& title, const std::string& xlabel, const std::string& ylabel) {
    const char* gnuplotCmd = "gnuplot -p";

    FILE* gnuplot = _popen(gnuplotCmd, "w");
    if (gnuplot) {
        
        fprintf(gnuplot, "set title '%s'\n", title.c_str());
        fprintf(gnuplot, "set xlabel '%s'\n", xlabel.c_str());
        fprintf(gnuplot, "set ylabel '%s'\n", ylabel.c_str());

        fprintf(gnuplot, "plot '-' with lines title 'Data'\n");
        for (size_t i = 0; i < x.size(); ++i) {
            fprintf(gnuplot, "%lf %lf\n", x[i], y[i]);
        }
        fprintf(gnuplot, "e\n");  
        fclose(gnuplot);
    }
    else {
        std::cerr << "Error opening pipe to gnuplot!" << std::endl;
        return -1;
    }

    return 0;
}



void test_speed(std::vector<std::vector<double>> input_matrix) {
    double err_exit = 0.01;
    double alpha = 5;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> weight;
    weight.resize(3);
    
    for (double i = 0.1; fabs(0.95 - i) > 1e-9; i += 0.01) {
        for (double& i : weight) i = 1;
        std::cout << "speed: " << i << std::endl;
        x.push_back(i);
        y.push_back(learn(weight, input_matrix, i, err_exit, alpha) + 1);
        std::cout << "speed " << i <<" end"<<std::endl;
    }
    plot(x, y,"dependence of the number of training generations on the learning rate value at constant values of error and slope coefficient","Speed","Num of epoch");
}


void test_alpha(std::vector<std::vector<double>> input_matrix) {
    double err_exit = 0.1;
    double learn_rate = 0.5;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> weight;
    weight.resize(3);
    for (double i = 0.1; fabs(6 - i) > 1e-9; i += 0.01) {
        std::cout << "alpha: " << i << std::endl;
        for (double& i : weight) i = 1;
        x.push_back(i);
        y.push_back(learn(weight, input_matrix, learn_rate, err_exit, i) + 1);
        std::cout << "alpha " << i << " end" << std::endl;
    }
    plot(x, y, "Dependence of the number of training generations on the slope coefficient at constant values of error and learning rate", "slope coefficient", "Num of epoch");
}


void test_error(std::vector<std::vector<double>> input_matrix) {
    double alpha = 1;
    double learn_rate = 0.2;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> weight;
    weight.resize(3);
    for (double i = 0.03; fabs(0.3- i) > 1e-9; i += 0.01) {
        std::cout << "error: " << i << std::endl;
        for (double& i : weight) i = 1;
        x.push_back(i);
        y.push_back(learn(weight, input_matrix, learn_rate, i, alpha) + 1);
        std::cout << "error " << i << " end" << std::endl;
    }
    plot(x, y, "Dependence of the number of training generations on the error value at constant values of the slope coefficient and learning rate", "Error", "Num of epoch");
}

std::vector<int> test_class(std::vector<std::vector<double>> res) {
    FILE* gp = _popen("gnuplot -p", "w");

    if (!gp) {
        std::cerr << "Error opening pipe to gnuplot!" << std::endl;
    }

    std::vector<double> x_class_1, y_class_1, x_class_2, y_class_2, x_err, y_err;
    std::vector<int> num_per_class(3, 0);

    for (const auto& row : res) {
        double feature_1 = row[0];
        double feature_2 = row[1];
        double classification = row[2];

        std::cout << "Expected: " << row[3] << " Reality: " << row[2] << '\n';
        if ((fabs(row[2] - row[3]) > 0.5) or (classification < 0.6 and classification > 0.4)) {
            x_err.push_back(feature_1);
            y_err.push_back(feature_2);
            ++num_per_class[2];

        }
        else if (classification > 0.6) {
            x_class_1.push_back(feature_1);
            y_class_1.push_back(feature_2);
            ++num_per_class[0];

        }
        else if(classification < 0.4){
            x_class_2.push_back(feature_1);
            y_class_2.push_back(feature_2);
            ++num_per_class[1];
        }
    }

    fprintf(gp, "set title 'Result of classification'\n");
    fprintf(gp, "set xlabel 'Sign 1'\n");
    fprintf(gp, "set ylabel 'Sign 2'\n");

    fprintf(gp, "plot '-' with points pointtype 7 linecolor rgb 'blue' title 'Class 1', ");
    fprintf(gp, "'-' with points pointtype 7 linecolor rgb 'red' title 'Class 0', ");
    fprintf(gp, "'-' with points pointtype 7 linecolor rgb 'green' title 'Error'\n");


    for (size_t i = 0; i < x_class_1.size(); ++i) {
        fprintf(gp, "%lf %lf\n", x_class_1[i], y_class_1[i]);
    }
    fprintf(gp, "e\n");


    for (size_t i = 0; i < x_class_2.size(); ++i) {
        fprintf(gp, "%lf %lf\n", x_class_2[i], y_class_2[i]);
    }
    fprintf(gp, "e\n");

    for (size_t i = 0; i < x_err.size(); ++i) {
        fprintf(gp, "%lf %lf\n", x_err[i], y_err[i]);
    }
    fprintf(gp, "e\n");

    fprintf(gp, "pause -1\n");
    fclose(gp);

    return num_per_class;
}




int main() {
    std::string path = "C:/Users/QSUS/Downloads/LAB1_1.xlsx";
    std::string path2 = "C:/Users/QSUS/Downloads/LAB1_2.xlsx";
    auto input_matrix = read_excel(path);
    auto temp = read_excel(path2);

    for (int i = temp.size() - 1; i > temp.size() - 1601; --i) {
        input_matrix.push_back(temp[i]);
        temp.pop_back();
    }

    for (int i = 0; i < 400; ++i) {
        input_matrix.push_back(temp[i]);
        std::swap(temp[i], temp[temp.size() - 1]);
        temp.pop_back();
    }
    




    std::vector<double> input_weight;
    double learn_rate = 0.5;
    double err_exit= 0.5;
    double alpha = 9;
    double result;
    int ep;

    test_speed(input_matrix);
    test_alpha(input_matrix);
    test_error(input_matrix);


    std::vector<std::string> lines;
    std::ifstream file("weight.txt");
    if (!file) {
        srand(time(nullptr));
        for (int i = 0; i < 3; ++i) {
            input_weight.push_back(rand() % 100 / 100);
        };
        std::cout<<learn(input_weight, input_matrix, learn_rate, err_exit, alpha)<<std::endl;
        
    }
    else {

        std::string line;
        int i = 0;
        while (getline(file, line)) {
            lines.push_back(line);
        }
    }

    file.close();
    std::string token;
    int i;
    for (i = 0; i < lines.size(); ++i) {
        std::istringstream stream(lines[i]);
        getline(stream, token, '\t');
        input_weight.push_back(stod(token));
    }
    


    std::vector<std::vector<double>> res_class;
    for (int i = 0; i < temp.size(); ++i) {
        res_class.push_back(std::vector<double> ());
        res_class[i].push_back(temp[i][0]);
        res_class[i].push_back(temp[i][1]);

        res_class[i].push_back(out_n(input_weight, temp[i], alpha));
        res_class[i].push_back(temp[i].back());
    }
    
    auto num = test_class(res_class);
    std::cout << "class 1: " << num[0] << "; class 0: " << num[1]<<" Error: "<<num[2]<<std::endl;


    std::ofstream file2("weight.txt");
    if (!file2) {
        std::cerr << "fatal error: cannot open file for writing!" << std::endl;
        exit(1); 
    }

    for (double& i : input_weight) {
       
        file2 << i;
        file2 << '\n';
    }

}
    