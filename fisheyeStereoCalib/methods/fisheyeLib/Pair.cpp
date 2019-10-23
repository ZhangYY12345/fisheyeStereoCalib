//
//  Pair.cpp
//  Calibration
//
//  Created by Ryohei Suda on 2014/06/02.
//  Copyright (c) 2014年 Ryohei Suda. All rights reserved.
//

#include "Pair.h"

//对所有点（图像上的点，图像像素坐标）计算其在相机坐标系的坐标
void Pair::calcM()
{
    for (int i = 0; i < 2; ++i) {
        for(auto &line : edge[i]) {
            for (auto &point : line) {
                point->calcM();
            }
        }
    }
}

//对边缘edge中的每条线line上的每个点point，求导
void Pair::calcMd()
{
    for (int i = 0; i < 2; ++i) {
        for(auto &line : edge[i]) {	
            for (auto &point : line) {
                point->calcDerivatives();
            }
        }
    }
}

//?
void Pair::calcNormal()
{
    for (int i = 0; i < 2; ++i) {
        normalVector[i].clear();
        normalValue[i].clear();
        
        for (auto &line : edge[i]) {
            cv::Mat Mk = cv::Mat::zeros(3, 3, CV_64F);
            for (auto &point : line) {
                cv::Mat m(point->m);
                Mk += m * m.t();	//海森矩阵，二阶导
            }
            cv::Mat eigenValues, eigenVectors;
            cv::eigen(Mk, eigenValues, eigenVectors);	//计算 每条直线的特征值和特征向量？
            normalVector[i].push_back(eigenVectors);	//特征向量
            normalValue[i].push_back(eigenValues);		//特征值
        }
    }
}

//?
void Pair::calcLine()
{
    for (int i = 0; i < 2; ++i) {
        
        cv::Mat Ng = cv::Mat::zeros(3, 3, CV_64F);
        for (auto &n : normalVector[i]) {
            cv::Mat nk = n.row(2); //？
            Ng += nk.t() * nk;
        }
        
        cv::Mat eigenValues, eigenVectors;
        cv::eigen(Ng, eigenValues, eigenVectors);	//计算 同一幅图上的所有直线的特征向量与特征值？
        lineVector[i] = eigenVectors;
        lineValue[i] = eigenValues;
    }
}

//?
void Pair::calcMc()
{
    for (int i = 0; i < 2; ++i) {
        Mc[i].clear();
        
        for (auto &line : edge[i]) {
            Pair::C c;
            for (auto &point : line) {
                cv::Mat m(point->m);
                for (int j = 0; j < IncidentVector::nparam; ++j) {
                    cv::Mat mc(point->derivatives[j]);
                    c.ms[j] += mc * m.t();
                }
            }
            for (int j = 0; j < IncidentVector::nparam; ++j) {
                c.ms[j] += c.ms[j].t();
            }
            
            Mc[i].push_back(c);
        }
    }
    
}

//?
void Pair::calcMcc()
{
    for (int i = 0; i < 2; ++i) {
        Mcc[i].clear();
        
        for (auto &line : edge[i]) {
            Pair::Cc cc;
            for (auto &point : line) {
                for (int j = 0; j < IncidentVector::nparam; ++j) {
                    cv::Mat mc1(point->derivatives[j]);
                    for(int l = 0; l < IncidentVector::nparam; ++l) {
                        cv::Mat mc2(point->derivatives[l]);
                        cc.ms[j][l] += mc1 * mc2.t();
                    }
                }
            }
            Mcc[i].push_back(cc);
        }
    }
}

void Pair::calcNc()
{
    
    for (int i = 0; i < 2; ++i) {
        Pair::C c;
        
        for (int j = 0; j < normalVector[i].size(); ++j) {
            cv::Mat nk1 = normalVector[i][j].row(0).t();
            cv::Mat nk2 = normalVector[i][j].row(1).t();
            cv::Mat nk = normalVector[i][j].row(2).t();
            double muk1 = normalValue[i][j].at<double>(0);
            double muk2 = normalValue[i][j].at<double>(1);
            double muk = normalValue[i][j].at<double>(2);
            
            for (int l = 0; l < IncidentVector::nparam; ++l) {
                cv::Mat mkc = Mc[i][j].ms[l];
                cv::Mat nkc = - ((nk1.dot(mkc*nk))/(muk1-muk) *nk1) - ((nk2.dot(mkc*nk)/(muk2-muk))*nk2);
                c.ms[l] += nkc * nk.t();
            }
        }
        
        for (int l = 0; l < IncidentVector::nparam; ++l) {
            c.ms[l] += c.ms[l].t();
        }
        
        Nc[i] = c;
    }
    
}

void Pair::calcNcc()
{
    for (int i = 0; i < 2; ++i) {
        Pair::Cc cc;
        
        for (int j = 0; j < normalVector[i].size(); ++j) {
            cv::Mat nk1 = normalVector[i][j].row(0).t();
            cv::Mat nk2 = normalVector[i][j].row(1).t();
            cv::Mat nk = normalVector[i][j].row(2).t();
            double muk1 = normalValue[i][j].at<double>(0);
            double muk2 = normalValue[i][j].at<double>(1);
            double muk = normalValue[i][j].at<double>(2);
            
            for (int l = 0; l < IncidentVector::nparam; ++l) {
                cv::Mat mkc1 = Mc[i][j].ms[l];
                cv::Mat nkc1 = - ((nk1.dot(mkc1*nk))/(muk1-muk) *nk1) - ((nk2.dot(mkc1*nk)/(muk2-muk))*nk2);
                for (int m = 0; m < IncidentVector::nparam; ++m) {
                    cv::Mat mkc2 = Mc[i][j].ms[m];
                    cv::Mat nkc2 = - ((nk1.dot(mkc2*nk))/(muk1-muk) *nk1) - ((nk2.dot(mkc2*nk)/(muk2-muk))*nk2);
                    cc.ms[l][m] += nkc1 * nkc2.t();
                    if (isnan(cc.ms[l][m].at<double>(0))) {
                        std::cout << cc.ms[l][m] << std::endl;
                        std::cout << nk1 << std::endl;
                        std::cout << nk2 << std::endl;
                        std::cout << nk << std::endl;
                        std::cout << muk1 << std::endl;
                        std::cout << muk2 << std::endl;
                        std::cout << muk << std::endl;
                        exit(99);
                    }
                }
            }
        }
        Ncc[i] = cc;
    }
}

void Pair::calcLc()
{
    
    for (int i = 0; i < 2; ++i) {
        Pair::C c;
        cv::Mat lg1 = lineVector[i].row(0).t();
        cv::Mat lg2 = lineVector[i].row(1).t();
        cv::Mat lg = lineVector[i].row(2).t();
        double mu1 = lineValue[i].at<double>(0);
        double mu2 = lineValue[i].at<double>(1);
        double mu = lineValue[i].at<double>(2);
            
        for (int k = 0; k < IncidentVector::nparam; ++k) {
            c.ms[k] = - (lg1.dot(Nc[i].ms[k]*lg)*lg1 / (mu1-mu)) - (lg2.dot(Nc[i].ms[k]*lg)*lg2 / (mu2-mu));
        }
        lc[i] = c;
    }
    
}

void Pair::calcDerivatives()
{
    calcMd();
    calcMc();
    calcMcc();
    calcNc();
    calcNcc();
    calcLc();
}