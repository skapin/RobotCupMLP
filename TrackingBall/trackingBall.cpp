#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <cstdio>
#include <ctime>


using namespace cv;
using namespace std;


int ballTracking(char*name)
{
    VideoCapture cap;
      cap.open(name);
      if(!cap.isOpened())  // check if we succeeded
	{
	  cout << "Unable to read specified video file" << endl;
	  return EXIT_FAILURE;
	}
      Mat edges;
      Mat edges2;
      cap >> edges;

      double fc;
      fc = cap.get(CV_CAP_PROP_FRAME_COUNT);


      namedWindow("video Kalman",1);
      namedWindow("video Barycentre",1);


      //Kalman initialisation
      KalmanFilter KF(4, 2, 0);
      KF.transitionMatrix = *(Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
      Mat_<float> measurement(2,1);
      measurement.setTo(Scalar(0));


      KF.statePre.at<float>(0) = 0;
      KF.statePre.at<float>(1) = 0;
      KF.statePre.at<float>(2) = 0;
      KF.statePre.at<float>(3) = 0;
      setIdentity(KF.measurementMatrix);
      setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
      setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
      setIdentity(KF.errorCovPost, Scalar::all(.1));


      while(1)
        {
          cap >> edges;
          if(edges.empty())
              break;
          edges.copyTo(edges2);
          Point bary;
          bary.x = 0;
          bary.y = 0;
          Point bary2;
          bary2.x = 0;
          bary2.y = 0;
          int n = 0;
          double x = 0;
          double y = 0;
          int n2 = 0;
          double x2 = 0;
          double y2 = 0;


          //Pixel detection and getting the barycenter
          for(int i=0;i<edges.rows;i++)
              for(int j=0;j<edges.cols;j++)
              {
                if(edges.at<Vec3b>(i,j)[2]<150 || edges.at<Vec3b>(i,j)[0]>50 || edges.at<Vec3b>(i,j)[1]>100)
                {
                    edges.at<Vec3b>(i,j)[0]=0;
                    edges.at<Vec3b>(i,j)[1]=0;
                    edges.at<Vec3b>(i,j)[2]=0;
                }
                else
                {
                    if(std::abs(j-edges.cols/4) < std::abs(j-3*edges.cols/4))
                    {
                        n++;
                        x += j;
                        y += i;
                    }
                    else
                    {
                        n2++;
                        x2 += j;
                        y2 += i;
                    }
                }

             }
          if(n != 0)
          {
            x = x/n;
            y = y/n;
          }
          if(n2 != 0)
          {
            x2 = x2/n2;
            y2 = y2/n2;
          }

          //drawing ball center

          if(n!=0 && n2!=0)
          {

          //if the two barycenter are too close, draw only one ball
          if(std::abs(x-x2) < (n+n2)/10 && std::abs(y-y2) <(n+n2)/10)
          {

              x = x*n + x2*n2;
              y = y*n +y2*n2;
              if(n!=0 || n2!=0)
              {
              x = x/(n+n2);
              y = y/(n+n2);
              bary.x = x;
              bary.y = y;

              }
              if(bary.x >= 20 && bary.y >=20)
                  circle(edges2, bary, abs(sqrt(n+n2)), Scalar(0,0,255), 2);
          }
          else // else draw two balls
          {
          bary.x = x;
          bary.y = y;

          bary2.x = x2;
          bary2.y = y2;
          if(bary.x >= 20 && bary.y >=20 && n>25)
            circle(edges2, bary, abs(sqrt(n)), Scalar(0,0,255), 2);
          if(bary2.x >= 20 && bary2.y >=20 && n2>25)
            circle(edges2, bary2, abs(sqrt(n2)), Scalar(0,0,255), 2);
           }
           }
          else
          {

              bary.x = x;
              bary.y = y;

            if(n2 != 0)
              {
                bary.x = x2;
                bary.y = y2;
                n = n2;
              }
            if(bary.x >= 20 && bary.y >=20 && n>25)
                circle(edges2, bary, abs(sqrt(n)), Scalar(0,0,255), 2);
          }


          // Kalman prediction
          KF.predict();

          //Getting the measurements from the barycenter
          if(bary.x >= 20 && bary.y >=20)
          {
           measurement(0) = bary.x;
           measurement(1) = bary.y;
          }


          // Correcting the predicted value with our measurement
          Mat estimated = KF.correct(measurement);
          Point statePt(estimated.at<float>(0),estimated.at<float>(1));
          //drawing result
          circle(edges, statePt, 2, Scalar(255,255,255), -1);


          //using dilate and erode to improve result display
          Mat element = getStructuringElement( MORPH_RECT,Size( 3, 3 ),Point( 2, 2 ) );
          dilate(edges, edges,element );
          erode(edges, edges,element );
          dilate(edges, edges,element );
          dilate(edges, edges,element );

          imshow("video Kalman", edges);
          imshow("video Barycentre", edges2);
          if(waitKey(30) >= 0) break;
        }
      return 0;

}


int main(int argc, char*argv[])
{
  if(argc != 2)
    {
      cout << "Usage : ./trackingBall path_to_video" << endl;
      return EXIT_FAILURE;
    }
  else
    {
      ballTracking(argv[1]);
      return EXIT_SUCCESS;
    }
}
