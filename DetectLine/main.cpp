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
#include <stack>

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
bool nearf(float a, float b, float delta){
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

void apply_and(Mat &mat1, Mat &mat2, Mat& output,  int treshold){
    int i,j, k;


    if(mat1.rows == mat2.rows && mat1.cols == mat2.cols){
        output = mat1.clone();
        for (i=0; i<mat2.rows;i++)
            for(j=0;j<mat2.cols;j++){
            int moyenne1 = (mat1.at<Vec3b>(i,j)[0] + mat1.at<Vec3b>(i,j)[1] + mat1.at<Vec3b>(i,j)[2])/3;
            int moyenne2 = (mat2.at<Vec3b>(i,j)[0] + mat2.at<Vec3b>(i,j)[1] + mat2.at<Vec3b>(i,j)[2])/3;
            if(near(moyenne1, 255, treshold) && near(moyenne2, 255, treshold))
            {
                for(k=0;k<3;k++)
                {
                    output.at<Vec3b>(i,j)[k] = 255;
                }
            }
            else
                for(k=0;k<3;k++)
                {
                output.at<Vec3b>(i,j)[k] = 0;
            }
        }
    }
}

void check_border_and_add( bool** V, stack<Point> border, Mat& img, int threshold, Point current_pixel, Point current_check )
{
    if(!V[current_check.x][current_check.y]){
        int current_moyenne = (img.at<Vec3b>(current_pixel)[0]+
                               img.at<Vec3b>(current_pixel)[1]+
                               img.at<Vec3b>(current_pixel)[2])/3;
        int current_check_moyenne = (img.at<Vec3b>(current_check)[0]+
                                     img.at<Vec3b>(current_check)[1]+
                                     img.at<Vec3b>(current_check)[2])/3;

        if(near(current_moyenne, current_check_moyenne, threshold)){
            V[current_check.x][ current_check.y] = true;
            border.push( current_check );
        }

    }

}

void region_grow(bool** V, stack<Point> border, Mat &img, int threshold){
    Point current_pixel, bottom_pixel, top_pixel, left_pixel, right_pixel;
    while(!border.empty())
    {
        current_pixel = border.top();
        border.pop();
        check_border_and_add( V, border, img, threshold, current_pixel, Point(current_pixel.x, current_pixel.y-1) );
        check_border_and_add( V, border, img, threshold, current_pixel, Point(current_pixel.x+1, current_pixel.y) );
        check_border_and_add( V, border, img, threshold, current_pixel,Point(current_pixel.x-1, current_pixel.y) );
        check_border_and_add( V, border, img, threshold, current_pixel, Point(current_pixel.x, current_pixel.y+1) );
    }

}

bool near_Vec3b(Vec3b a, Vec3b b, int threshold)
{
    return (near(a[0],b[0], threshold) && near(a[1],b[1], threshold) && near(a[2],b[2], threshold));
}

bool near_Vec3b_3channel(Vec3b a, Vec3b b, int c1, int c2, int c3)
{
    return (near(a[0],b[0], c1) && near(a[1],b[1], c2) && near(a[2],b[2], c3));
}

bool near_Vec3b_3channel_ratio(Vec3b a, Vec3b b, float c1, float c2, float c3)
{
    return ( nearf (a[0]/(a[1]+1), b[0]/(b[1]+1),c1) &&  nearf (a[0]/(a[2]+1), b[0]/(b[2]+1),c2) &&   nearf (a[2]/(a[1]+1), b[2]/(b[1]+1),c3) );
}

bool near_Vec3b_3channel_sub(Vec3b a, Vec3b b, float c1, float c2, float c3)
{
    return ( nearf( abs(a[0]-a[1]), abs(b[0]-b[1]),c1) &&  nearf( abs(a[0]-a[2]), abs(b[0]-b[2]),c1) && nearf( abs(a[2]-a[1]), abs(b[2]-b[1]),c1) );
}



Point search_exclude (vector<bool**> regions, Mat& img, int threshold_search, Vec3b color){
    int i,j,k;
    bool ** excluded = new bool*[img.rows];
    for ( i = 0 ; i < img.rows ; ++i)
    {
        excluded[i] = new bool[img.cols];
    }
    for(i=0;i<regions.size();i++)
    {
        for(j=0;j<img.rows;j++)
            for(k=0;k<img.cols;k++)
                excluded[j][k] += regions[i][j][k];
    }
    for(i=0;i<img.rows;i++)
        for(j=0; j<img.cols;j++){
        if(!excluded[i][j] && near_Vec3b(img.at<Vec3b>(i,j), color, threshold_search))
            return Point(i,j);
    }
    return Point(-1,-1);
}

void detect_regions(vector<bool**>& regions, vector< stack<Point> > borders, Mat& img, int threshold, int threshold_search, int min_region_size, Vec3b color){

    int i,j;
    Point region_starting_point = search_exclude(regions, img, threshold_search, color);
    bool ** region= new bool*[img.rows];
    for ( i = 0 ; i < img.rows ; ++i)
    {
        region[i] = new bool[img.cols];
    }
    stack<Point> border;

    while( region_starting_point != Point(-1,-1))
    {
        region= new bool*[img.rows];
        for ( i = 0 ; i < img.rows ; ++i)
        {
            region[i] = new bool[img.cols];
        }
        for(i=0;i<img.rows;i++)
            for(j=0; j<img.cols; j++)
            {
            region[i][j] = false;
        }
        border =  stack<Point>() ;
        region[region_starting_point.x][region_starting_point.y] = true;
        border.push(region_starting_point);
        region_grow(region, border, img, threshold);
        int region_size = 0;
        for(i=0; i<img.rows; i++)
            for(j=0;j<img.cols; j++)
                if (region[i,j])
                    region_size++;

        if( region_size >= min_region_size)
        {
            regions.push_back( region );
            borders.push_back( border );
        }
        region_starting_point = search_exclude(regions, img, threshold_search, color);
    }

}

void show_regions (Mat& input, Mat& output, int threshold, int thresold_search, int min_region_size, Vec3b color, Vec3b region_color)
{
    output = input.clone();
    vector<bool**> regions ;
    vector<stack<Point> > borders;
    detect_regions(regions, borders, output, threshold, thresold_search, min_region_size, color);

    int i,j, k;
    for(i=0;i<output.rows;i++)
        for(j=0;j<output.cols; j++)
            for(k=0;k<3; k++)
                output.at<Vec3b>(i,j)[k]=0;

    for(i=0; i< regions.size(); i++)
        for(j=0; j< output.rows;j++)
            for(k=0; k<output.cols;k++)
                if(regions[i][j][k])
                    output.at<Vec3b>(j,k)= region_color;

}

Vec3b get_average( Mat& input2, int x, int y )
{
    Mat input = input2.clone();
    Vec3b moy;
    int a=0,b=0,c=0;
    a = (int)(input.at<Vec3b>(y,x)[0]);
    b = (int)(input.at<Vec3b>(y,x)[1]);
    c = (int)(input.at<Vec3b>(y,x)[2]);

    a = a +(int)(input.at<Vec3b>(y+1,x)[0]);
    b = b +(int)(input.at<Vec3b>(y+1,x)[1]);
    c = c + (int)(input.at<Vec3b>(y+1,x)[2]);



    moy[0] = a/2.0;
    moy[1] = b/2.0;
    moy[2] = c/2.0;

    return moy;
}

bool dist_ok( Point a, Point b, int distance)
{
    int k =(a.x -b.x)*(a.x-b.x) + (a.y - b.y)*(a.y-b.y) ;
    return (k< (distance*distance) );
}

void draw_affine( Mat& input, Mat& output, vector<Point> ptns, Scalar color, int decalx )
{
    output = input.clone();
    int i;
    float a=0.0, atemp;
    int start = 0, nb_points = 0;
    bool valid, dist_valid;
    int t;
    int end=0;

    for(i=0; i<ptns.size()-1; i++)
    {
        t = 1;
        valid = false;
        dist_valid = false;
        while(!valid && t < 4 && (i+t) < ptns.size())
        {
            Point next_point = ptns[i+t];

           if (dist_ok(ptns[i+t], ptns[i], 2*decalx))
            {
                atemp = ((float)(next_point).y - (ptns[start]).y) /((next_point).x - (ptns[start]).x);
                a = ((nb_points)*a + atemp)/(nb_points+1);
                dist_valid = true;

            }
            if ((a == 0 && dist_valid == true) || (atemp/a >0.85 && atemp/a <1.15) )
            {
                valid = true;
            }
            t++;
        }

        if(valid)
        {
            end = i+t-1;
            nb_points ++;
        }
        if(!valid )
        {
            if( nb_points > 0)
                line(output, ptns[start], ptns[end], Scalar(0,i*8,0),2);
            nb_points = 0;
            a =0;
            end = i;
            start = i+1;
        }
    }
    line(output, ptns[start], ptns[end], Scalar(0,250,0),2);
    












    /*  Point next_point;
    Point previous_valid_point = ptns[0];
    for(i=0;(ptns.size() != 0 ) && (i<ptns.size()-1);i++)
    {
        bool valid = false;
        int t=1;

        while ( i+t < ptns.size() && !dist_ok(ptns[i], ptns[i+t], 2*decalx)  )
        {
            valid = true;
            t++;
        }
        if ( valid )
        {
            next_point = ptns[i+t];
            nbpoints++;
            atemp = ((float)(next_point).y - (ptns[i]).y) /((next_point).x - (ptns[i]).x);
            if(a == 0 || (atemp/a < 1.2 && atemp/a >0.8))
            {
                previous_valid_point = next_point;
                a = ((nbpoints -1)*a + atemp)/(nbpoints);
            }

        else if( nbpoints > 0)
        {
            cout<<"*"<<endl;
            line(output, ptns[start], previous_valid_point, color, 5);
        }
        a = 0;
        start = i;
        nbpoints = 0;
        previous_valid_point = ptns[start];


    }*/
 }

bool light_near(Vec3b a, Vec3b b, Vec3b a_perfect, Vec3b b_perfect, float delta)
{
    float aperfect_scale_bperfect = (((float)a_perfect[0]) + a_perfect[1] + a_perfect[2])/(b_perfect[0]+b_perfect[1]+b_perfect[2]);
    float a_scale_b = ((((float)(a[0])) + a[1] + a[2]) / (b[0] + b[1] + b[2]));

    return (1.-delta <( (a_scale_b/aperfect_scale_bperfect)) && (1.+delta > (a_scale_b/aperfect_scale_bperfect)));
}



#define VERT 0x00000001
#define BLANC 0x00000002
#define  UNK 0x00000004
void detectLinePoints (Mat& input, Vec3b color1, Vec3b color2, int slices){
    int i,x=input.cols - 10,y=input.rows-10, decalx = input.cols/slices ;
    int state = 0, previous_state = 0;
    Mat img_out = input.clone();
    Mat tmp;
    Mat out;
    /*  Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 10, 10) );
    erode( img_out, out, element );
    dilate( out, img_out, element );
*/
    
    vector<Point> ptns;
    Vec3b green_found;
    Vec3b white_found;

    imshow("Algo", img_out);
    for(i=0; i<slices; i++){
        state = 0;
        y = input.rows-10;
        while ( y > 10 && state != UNK )
        {
            previous_state = state;
            state = 0;
            Vec3b color = input.at<Vec3b>(y,x);
            /*          color[0] = color[0]/2;
            color[1] = color[1]/2;
            color[2] = color[2]/2;

            color[0] = (int)color[0] + (int)(input.at<Vec3b>(y+1,x)[0])/2;
            color[1] = (int)color[0] + (int)(input.at<Vec3b>(y+1,x)[1])/2;
            color[2] = (int)color[0] +(int)(input.at<Vec3b>(y+1,x)[2]/2);
*/
            // rectangle(img_out,Rect( Point(x-4,y),Point(x+4,y+1) ),Scalar(10,10,10),1);
            /**
              * Param  80,80 = near_Vec3b
              *
              **/
            // if(near_Vec3b_3channel_ratio(color1, color,0.30,0.30,0.30) && near_Vec3b_3channel(color1, color,80,80,80))
            if(near_Vec3b_3channel_sub(color1, color,70,30,30) && near_Vec3b_3channel(color1, color,120,120,120))
            {
                if(previous_state == VERT)
                    state = 3;
                else
                    state=BLANC;
                white_found = color;
                rectangle(img_out,Rect( Point(x-4,y),Point(x+4,y+1) ),Scalar(250,250,250),1);
            }
            //if (near_Vec3b_3channel_ratio(color2, color,1.0,1.00,1.00) /*&& near_Vec3b_3channel(color2, color,100,100,100)*/ )
            if (near_Vec3b_3channel_sub(color2, color,50,10,10) && near_Vec3b_3channel(color2, color,80,80,80) )
            {
                if(previous_state == BLANC /*|| state == BLANC*/)
                    state = 3;
                else
                    state=VERT;
                green_found = color;
                rectangle(img_out,Rect( Point(x-4,y),Point(x+4,y+1) ),Scalar(0,250,0),1);
            }

            if(state == 0 && previous_state != 0 )
            {
                rectangle(img_out,Rect( Point(x-6,y),Point(x+6,y+1) ),Scalar(0,0,250),1);
                green_found = NULL;
                white_found = NULL;
            }

            if((state ==3)/* && (light_near (green_found, white_found, color2, color1, 0.2))*/)
            {
                ptns.push_back(Point(x,y));
                rectangle(img_out,Rect( Point(x-4,y-4),Point(x+4,y+4) ),Scalar(250,0,0),3);
                state = UNK;
            }



            /*else if ( state != 0  )
            {
                state = 0;
                rectangle(img_out,Rect( Point(x-6,y+1),Point(x+6,y+2) ),Scalar(0,0,250),1);
            }

            //on est passé sur du blanc et du vert, alors on fait le traitement du point
            if ( (state & VERT)  && (state & BLANC) )
            {
                //process
                state = UNK;
                ptns.push_back(Point(x,y));
                rectangle(img_out,Rect( Point(x-4,y-4),Point(x+4,y+4) ),Scalar(250,0,0),3);
            }*/

            y--;
        }
        x-= decalx;
    }
    draw_affine(img_out, tmp, ptns, Scalar(0,0,230), decalx);
    img_out= tmp;
    imshow("Algo", img_out);

    
    cout<<"rofl"<<endl;
}


void detectLinePoints_glitch(Mat& input, Vec3b color1, Vec3b color2, int slices){
    int i,x=6,y=6, decalx = input.cols/slices ;
    int state = 0, previous_state = 0;
    Mat img_out = input.clone();
    vector<Point> ptns;
    Vec3b green_found;

    Vec3b white_found;

    imshow("Algo", img_out);
    for(i=0; i<slices; i++){
        state = 0;
        y = 6;
        while ( y < input.rows-6 && state != UNK )
        {
            previous_state = state;
            state = 0;
            Vec3b color = input.at<Vec3b>(y,x);
            /*          color[0] = color[0]/2;
            color[1] = color[1]/2;
            color[2] = color[2]/2;

            color[0] = (int)color[0] + (int)(input.at<Vec3b>(y+1,x)[0])/2;
            color[1] = (int)color[0] + (int)(input.at<Vec3b>(y+1,x)[1])/2;
            color[2] = (int)color[0] +(int)(input.at<Vec3b>(y+1,x)[2]/2);
*/
            rectangle(img_out,Rect( Point(x-4,y),Point(x+4,y+1) ),Scalar(10,10,10),1);
            /**
              * Param  80,80 = near_Vec3b
              *
              **/
            // if(near_Vec3b_3channel_ratio(color1, color,0.30,0.30,0.30) && near_Vec3b_3channel(color1, color,80,80,80))
            if(near_Vec3b_3channel_sub(color1, color,55,30,30) && near_Vec3b_3channel(color1, color,120,120,120))
            {
                if(previous_state == VERT)
                    state = 3;
                else
                    state=BLANC;
                white_found = color;
                rectangle(img_out,Rect( Point(x-4,y),Point(x+4,y+1) ),Scalar(250,250,250),1);
            }
            //if (near_Vec3b_3channel_ratio(color2, color,1.0,1.00,1.00) /*&& near_Vec3b_3channel(color2, color,100,100,100)*/ )
            if (near_Vec3b_3channel_sub(color2, color,50,50,20) && near_Vec3b_3channel(color2, color,60,60,60) )
            {
                if(previous_state == BLANC || state == BLANC)
                    state = 3;
                else
                    state=VERT;
                green_found = color;
                rectangle(img_out,Rect( Point(x-4,y),Point(x+4,y+1) ),Scalar(0,250,0),1);
            }

            if((state ==3)/* && (light_near (green_found, white_found, color2, color1, 0.2))*/)
            {
                ptns.push_back(Point(x,y));
                //rectangle(img_out,Rect( Point(x-4,y-4),Point(x+4,y+4) ),Scalar(250,0,0),3);
                state = 0;
            }

            if(state == 0 && previous_state != 0 )
            {
                rectangle(img_out,Rect( Point(x-6,y),Point(x+6,y+1) ),Scalar(0,0,250),1);
                green_found = NULL;
                white_found = NULL;
            }

            /*else if ( state != 0  )
            {
                state = 0;
                rectangle(img_out,Rect( Point(x-6,y+1),Point(x+6,y+2) ),Scalar(0,0,250),1);
            }

            //on est passé sur du blanc et du vert, alors on fait le traitement du point
            if ( (state & VERT)  && (state & BLANC) )
            {
                //process
                state = UNK;
                ptns.push_back(Point(x,y));
                rectangle(img_out,Rect( Point(x-4,y-4),Point(x+4,y+4) ),Scalar(250,0,0),3);
            }*/

            y++;
        }
        x+= decalx;
    }
    //  draw_affine(img_out, img_out, ptns, Scalar(0,0,230));
    imshow("Algocheated", img_out);
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

        Mat src, src_gray, src_tmp;
        Mat grad,canny;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;

        /// Generate grad_x and grad_y
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;

        *video >> current_img ;
        //HSV
        //cvtColor( current_img, src_gray, CV_BGR2RGB);// on convertie

        //imshow("And", current_img);

        // detectLinePoints(current_img,Vec3b(253,253,233),Vec3b(122,212,137),10);
        detectLinePoints_glitch(current_img,Vec3b(230,230,230),Vec3b(140,180,120), 56);
        detectLinePoints(current_img,Vec3b(230,230,230),Vec3b(140,180,120), 30);

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

        /*  /// Gradient X
        //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );
        
        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );
        
        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        
        //threshold( grad, grad, 35,255,CV_THRESH_BINARY);

        



        // current_img_histo = *(applyHistogram(current_img_lab));
        Mat output = *(assHole(grad, 60, 90, 256));

        Mat output2 = *(assHole(src_gray, 30, 150, 256));
        int dilation_size = 0;
        Mat element = getStructuringElement(  MORPH_RECT,
                                             Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                             Point( dilation_size, dilation_size ) );
        erode( output, output2, element );
        dilate( output2, output2, element );
        
        Mat output3;

        show_regions(src_gray, output3, 40,5,50,Vec3b(115,206,130), Vec3b(255,255,255));

       // Mat result_and;
     //   apply_and(output2,output, result_and, 150);
        imshow("Sobel", grad);
        imshow("Trololo", output3);
        imshow("Sobel+Seuil", output);
        imshow("Seuil+dilate+sobel", output2);
        //imshow("And", result_and);*/
        waitKey(30);
    }

    return 0;

}
