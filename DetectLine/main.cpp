/**
  * @author Quentin MAOUHOUB, Florian BOUDINET, Groupe MLP, Binome 2
  * @since Decembre 2012
  * @file main.cpp
  *
  **/

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

/**
  * Ouvre la video dont le lien est passé en parametre
  *
  **/
VideoCapture* openVideo( char* video_path )
{
    VideoCapture* video = new VideoCapture( video_path );
    if(!video->isOpened())
    {
        cout<<"ERROR OPENING VIDEO, BYE BYE"<<endl;
        return NULL;
    }
    return video;

}
/**
  * affiche quelques informations sur la video passée en parametre.
  **/
void printVideoInformation(VideoCapture* video)
{
    cout<<"===== VIDEO Information=========="<<endl;
    cout<<"FPS : "<<video->get(CV_CAP_PROP_FPS )<<endl;
    cout<<"FRAME COUNT : "<<video->get(CV_CAP_PROP_FRAME_COUNT )<<endl;
    cout<<"FRAME SIZE : "<<video->get(CV_CAP_PROP_FRAME_WIDTH )<<"x"<<video->get(CV_CAP_PROP_FRAME_HEIGHT )<<endl;

}



int H_function( int intensity, int Imax, int nb_pixels, int* histogram)
{
    return (int)(Imax * (float)((float)histogram[intensity]/(float)nb_pixels));
}

Mat * applyHistogram(Mat& img){

    Mat* new_Img = new Mat();
    *new_Img=img.clone();
    int histogram[img.channels()][256];
    int Imax[img.channels()] ;

    //INITIALISATION
    for (int k = 0; k<256; k++)
    {
        for(int i=0;i<img.channels();i++)
            histogram[i][k] = 0;
    }

    for (int k = 0; k<img.channels(); k++)
    {
        Imax[k] = 0;
    }

    //CREATION HISTOGRAMME
    for (int i = 0; i<img.rows;i++)
        for(int j = 0; j < img.cols; j++)
            for(int channel = 0; channel <img.channels();channel++)
            {
        histogram[channel][img.at<Vec3b>(i,j)[channel]]++;

        if (Imax[channel] < img.at<Vec3b>(i,j)[channel])
            Imax[channel] = img.at<Vec3b>(i,j)[channel];
    }
    for(int i=0;i<img.channels();i++)
        for(int  j=1;j<256;j++)
        {
        histogram[i][j] += histogram[i][j-1];
    }

    int nb_pixels = img.rows * img.cols;

    //CREATION IMAGE
    for (int i = 0; i<img.rows;i++)
        for(int j = 0; j < img.cols; j++)
            for(int channel = 0; channel <img.channels();channel++)
            {
        new_Img->at<Vec3b>(i,j)[channel] = H_function(img.at<Vec3b>(i,j)[channel], 255, nb_pixels, histogram[channel]);

    }

    return new_Img;

}

Mat* toGray(const Mat& img, int channel )
{
    Mat* output = new Mat(img.rows, img.cols, CV_8UC3);
    for (int i = 0; i<img.rows;i++)
        for(int j = 0; j < img.cols; j++)
        {
        output->at<Vec3b>(i,j)[0] = (img.at<Vec3b>(i,j)[channel]);
    }
    return output;
}

bool near(int a, int b, int delta){
    if(a<b)
        return (a+delta>b);
    else
        return (a-delta <b);
}

Mat * assHole(Mat& img, int delta, int min, int max){
    Mat * new_Img = new Mat();
    *new_Img=img.clone();

    for(int i =0; i< img.rows;i++)
    {
        for(int j = 0; j< img.cols; j++)
        {
            int moyenne = (img.at<Vec3b>(i,j)[0] + img.at<Vec3b>(i,j)[1] + img.at<Vec3b>(i,j)[2])/3;
            if( moyenne> min && moyenne <max && near(img.at<Vec3b>(i,j)[0],moyenne, delta) && near(img.at<Vec3b>(i,j)[1],moyenne, delta) && near(img.at<Vec3b>(i,j)[2],moyenne, delta))
                for(int k=0; k<img.channels();k++)
                    new_Img->at<Vec3b>(i,j)[k] = img.at<Vec3b>(i,j)[k];
            else
                for(int k=0; k<img.channels();k++)
                    new_Img->at<Vec3b>(i,j)[k] = 0;
        }
    }
    return new_Img;
}

int main(int argc, char** argv)
{
    bool run = true;
    Mat current_img, current_img_lab, current_img_histo;

    namedWindow("edges",1);



    // Verification des arguments
    if ( argc < 2 )
    {
        cout<<"Error, you need to give video PATH"<<endl;
        return 1;
    }

    VideoCapture* video = openVideo( argv[1] );
    printVideoInformation( video );

    vector<Mat> planes;

    Mat c1, c2, c3;

    while( run )
    {
        *video >> current_img ;
        //HSV
        cvtColor( current_img, current_img_lab, CV_BGR2RGB);// on convertie


        //   Mat* output = toGray(  current_img_lab, 1 );
        // cvtColor( *output, current_img_lab, CV_BGR2GRAY );// on convertie


        /*split(current_img_lab, planes);
        vector<Mat> results = planes;

        //equalizeHist(planes[0], results[0]);
        //equalizeHist(planes[1], results[1]);
        //equalizeHist(planes[2], results[2]);
        threshold(planes[0],results[0],100,200,ADAPTIVE_THRESH_MEAN_C);
        threshold(planes[1],results[1],205,255,ADAPTIVE_THRESH_MEAN_C);
        threshold(planes[2],results[2],100,200,ADAPTIVE_THRESH_MEAN_C);

        merge(results, current_img_lab);
*/
        Mat output = *(assHole(current_img_lab, 40, 180, 256));




        // current_img_histo = *(applyHistogram(current_img_lab));
        imshow("edges", output);
        waitKey(30);
    }

    return 0;

}
