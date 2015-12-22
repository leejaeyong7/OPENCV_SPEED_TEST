/*******************************************************
/			SIFT METHOD Test							/
/			PURE Research Project Test # 2				/
/			Jae Yong Lee								/
********************************************************/
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "timer.h"
using namespace cv;
using namespace std;

double test_fm(char* imgFile, char* videoFile);
void test_od(char* imgFile, char* videoFile);
void test_ev(char* imgFile, char* videoFile);


/** @function main */
int main( int argc, char** argv )
{
    //------------------------------------------------------------//
    //  choose test environment
    //------------------------------------------------------------//
    if(argc!=3)
    {
        cout<<argc<<endl;
        cout<<"invalid input values"<<endl;
        return -1;
    }
    
    int t_choose = 0;
    
    for(;;)
    {
        while(t_choose<1 || t_choose>4)
        {
            cout<<"<   Test environment setting   >"<<endl;
            cout<<"   --------------------------   "<<endl;
            cout<<"|  Choose test environment:    |"<<endl;
            cout<<"|  1. test feature matching    |"<<endl;
            cout<<"|  2. test object detection    |"<<endl;
            cout<<"|  3. test everything possible |"<<endl;
            cout<<"|  4. exit                     |"<<endl;
            cout<<"   --------------------------   "<<endl;
            cout<<"  I choose... : ";
            cin>>t_choose;
        }
        switch(t_choose)
        {
            case 1:
            {
                t_choose = 0;
                test_fm(argv[1], argv[2]);
                break;
            }
            case 2:
            {
                t_choose = 0;
                test_od(argv[1], argv[2]);
                break;
            }
            case 3:
            {
                t_choose = 0;
                test_ev(argv[1], argv[2]);
                break;
            }
            case 4:
            {
                return 0;
            }
            
        }
    }
    return 0;
}

    //------------------------------------------------------------//
    //  setting the environment
    //------------------------------------------------------------//

    
    //------------------------------------------------------------//
    //  testing feature detector
    //  test feature detecting environment
    //  returns keypoints
    //------------------------------------------------------------//


double test_fm(char* imgFile, char* videoFile)
{
    //declare variables
    cout<<endl;
    cout<<"Testing feature detector"<<endl;
    
    //read image/movie from file
    Mat img = imread(imgFile, CV_LOAD_IMAGE_GRAYSCALE);
    VideoCapture cap(videoFile);
    Mat video;
    
    //initialize choosing variables
    int f_choose = 0;
    int d_choose = 0;
    
    //set frame to 0;
    int frame = 0;
    
    
    //interface for feature detector
    while(f_choose<1 || f_choose > 6)
    {
        cout<<"<   Feature Detector setting   >"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"|  Choose feature detector:    |"<<endl;
        cout<<"|  1. SIFT                     |"<<endl;
        cout<<"|  2. SURF                     |"<<endl;
        cout<<"|  3. ORB                      |"<<endl;
        cout<<"|  4. BRISK                    |"<<endl;
        cout<<"|  5. FAST                     |"<<endl;
        cout<<"|  6. exit                     |"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"  I choose... : ";
        cin>>f_choose;
    }
    
    //interface for descriptor extractor
    while(d_choose<1 || d_choose > 6)
    {
        cout<<"< Descriptor Extractor setting >"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"|  Choose Descriptor Extractor:|"<<endl;
        cout<<"|  1. SIFT                     |"<<endl;
        cout<<"|  2. SURF                     |"<<endl;
        cout<<"|  3. ORB                      |"<<endl;
        cout<<"|  4. BRISK                    |"<<endl;
        cout<<"|  5. FREAK                    |"<<endl;
        cout<<"|  6. exit                     |"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"  I choose... : ";
        cin>>d_choose;
    }
    
    
    //start timer
    Time timer;
    for(;;)
    {
        //capture frame from video
        cap>>video;
        
        //set default virtual function for descriptor, extractor
        Ptr<FeatureDetector> descriptor;
        Ptr<DescriptorExtractor> extractor;
        
        //set blank keypoint vector
        vector<KeyPoint> KP1;
        vector<KeyPoint> KP2;
        
        //set blank Matrix for computed img, frame
        Mat computedimg;
        Mat computedcap;
        
        //set feature detector and descriptor extractor
        switch(f_choose)
        {
            case 1:
            {
                descriptor = FeatureDetector::create("SIFT");
                break;
            }
            case 2:
            {
                descriptor = FeatureDetector::create("SURF");
                break;
            }
            case 3:
            {
                descriptor = FeatureDetector::create("ORB");
                break;
            }
            case 4:
            {
                descriptor = FeatureDetector::create("BRISK");
                break;
            }
            case 5:
            {
                descriptor = FeatureDetector::create("FAST");
                break;
            }
            case 6:
            {
                return 0;
                break;
            }
        }
        switch(d_choose)
        {
            case 1:
            {
                extractor = DescriptorExtractor::create("SIFT");
                break;
            }
            case 2:
            {
                extractor = DescriptorExtractor::create("SURF");
                break;
            }
            case 3:
            {
                extractor = DescriptorExtractor::create("ORB");
                break;
            }
            case 4:
            {
                extractor = DescriptorExtractor::create("BRISK");
                break;
            }
            case 5:
            {
                extractor = DescriptorExtractor::create("FREAK");
                break;
            }
            case 6:
            {
                return 0;
                break;
            }
        }
        
        //get keypoint from descriptor
        descriptor->detect(img, KP1);
        descriptor->detect(video, KP2);
        
        //compute keypoint from extractor
        extractor->compute(img, KP1, computedimg);
        extractor->compute(img, KP2, computedcap);
        
        
        //match following points
        
        BFMatcher Matcher(NORM_L2);
        //FlannBasedMatcher Matcher;
        vector<DMatch> matches;
        
        Matcher.match(computedimg, computedcap, matches);
        
        //draw final image with all variables and keypoints
        Mat final;
        drawMatches(img, KP1, video, KP2, matches,final);
        imshow("OPENCV TEST", final);
        
        //add frame
        frame++;
        
        //exit if pressed key
        if(waitKey(30) >=0)
            break;
    }
    
    // write time record of images
    double seconds = timer.elapsedTimeSeconds();
    cout<<"-----------------------------"<<endl;
    cout<<"Total Frame in movie : "<<frame<<endl;
    cout<<"Total elapsed time   : "<<seconds<<endl;
    cout<<"Frame per second     : "<<frame/seconds<<endl;
    cout<<"Original FPS of video: "<<30<<endl;
    cout<<"-----------------------------"<<endl;

    return frame/seconds;
}

    //------------------------------------------------------------//
    //  testing object detection
    //------------------------------------------------------------//
    

void test_od(char* imgFile, char* videoFile)
{
    //declare variables
    cout<<endl;
    cout<<"Testing feature detector"<<endl;
    
    //read image/movie from file
    Mat img = imread(imgFile, CV_LOAD_IMAGE_GRAYSCALE);
    VideoCapture cap(videoFile);
    //VideoCapture cap(0);
    Mat video;
    
    //initialize choosing variables
    int f_choose = 0;
    int d_choose = 0;
    
    //set frame to 0;
    int frame = 0;
    int dframe = 0;
    int ndframe = 0;
    
    //set angle and length saving variable
    double angle1=999;
    double angle2=999;
    double angle3=999;
    double angle4=999;
    
    double length1 = 0;
    double length2 = 0;
    double length3 = 0;
    double length4 = 0;
    
    //interface for feature detector
    while(f_choose<1 || f_choose > 6)
    {
        cout<<"<   Feature Detector setting   >"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"|  Choose feature detector:    |"<<endl;
        cout<<"|  1. SIFT                     |"<<endl;
        cout<<"|  2. SURF                     |"<<endl;
        cout<<"|  3. ORB                      |"<<endl;
        cout<<"|  4. BRISK                    |"<<endl;
        cout<<"|  5. FAST                     |"<<endl;
        cout<<"|  6. exit                     |"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"  I choose... : ";
        cin>>f_choose;
    }
    
    //interface for descriptor extractor
    while(d_choose<1 || d_choose > 6)
    {
        cout<<"< Descriptor Extractor setting >"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"|  Choose Descriptor Extractor:|"<<endl;
        cout<<"|  1. SIFT                     |"<<endl;
        cout<<"|  2. SURF                     |"<<endl;
        cout<<"|  3. ORB                      |"<<endl;
        cout<<"|  4. BRISK                    |"<<endl;
        cout<<"|  5. FREAK                    |"<<endl;
        cout<<"|  6. exit                     |"<<endl;
        cout<<"   --------------------------   "<<endl;
        cout<<"  I choose... : ";
        cin>>d_choose;
    }
    
    
    //start timer
    Time timer;
    for(;;)
    {
        //capture frame from video
        cap>>video;
        
        //set default virtual function for descriptor, extractor
        Ptr<FeatureDetector> descriptor;
        Ptr<DescriptorExtractor> extractor;
        
        //set blank keypoint vector
        vector<KeyPoint> KP1;
        vector<KeyPoint> KP2;
        
        //set blank Matrix for computed img, frame
        Mat computedimg;
        Mat computedcap;
        
        //set feature detector and descriptor extractor
        switch(f_choose)
        {
            case 1:
            {
                descriptor = Ptr<FeatureDetector>(new SiftFeatureDetector(500));
                break;
            }
            case 2:
            {
                descriptor = Ptr<FeatureDetector>(new SurfFeatureDetector(300));
                break;
            }
            case 3:
            {
                descriptor = FeatureDetector::create("ORB");
                break;
            }
            case 4:
            {
                descriptor = FeatureDetector::create("BRISK");
                break;
            }
            case 5:
            {
                descriptor = FeatureDetector::create("FAST");
                break;
            }
            case 6:
            {
                return;
                break;
            }
        }
        switch(d_choose)
        {
            case 1:
            {
                extractor = DescriptorExtractor::create("SIFT");
                break;
            }
            case 2:
            {
                extractor = DescriptorExtractor::create("SURF");
                break;
            }
            case 3:
            {
                extractor = DescriptorExtractor::create("ORB");
                break;
            }
            case 4:
            {
                extractor = DescriptorExtractor::create("BRISK");
                break;
            }
            case 5:
            {
                extractor = DescriptorExtractor::create("FREAK");
                break;
            }
            case 6:
            {
                return;
                break;
            }
        }
        
        //get keypoint from descriptor
        descriptor->detect(img, KP1);
        descriptor->detect(video, KP2);
        
        //compute keypoint from extractor
        extractor->compute(img, KP1, computedimg);
        extractor->compute(video, KP2, computedcap);
        
        
        //match following points
        //FlannBasedMatcher Matcher;
        BFMatcher Matcher(NORM_L2);
        vector<DMatch> matches;
        
        Matcher.match(computedimg, computedcap, matches);
        
		//calculate max and minimum value of surf/sift match's distance
		double max= 0; double min = 10000;
        
		for( int i = 0; i < computedimg.rows; i++ )
		{
            double dist = matches[i].distance;
            if( dist < min ) min = dist;
            if( dist > max ) max = dist;
		}
		//for values in between minimum and maximum, save it in vector 'good'
		std::vector< DMatch > good;
        
		for( int i = 0; i < computedimg.rows; i++ )
		{
            if( matches[i].distance <= 3*min+5 )
            {
                good.push_back( matches[i]);
            }
		}
		
		//write all image with object image and webcam to final image
		Mat final;
		drawMatches(img, KP1, video, KP2, good, final, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        //cout<<min<<"   "<<good.size()<<endl;
        
        if(good.size()>4 )
        {
            
        
            //calculate corner points of image and draw a line on the final image
            std::vector<Point2f> obj;
            std::vector<Point2f> scene;
        
            for( int i = 0; i < good.size(); i++ )
            {
                obj.push_back( KP1[ good[i].queryIdx ].pt );
                scene.push_back( KP2[ good[i].trainIdx ].pt );
            }
            // calculate the corner regarding its shift
            // automorphs 4 corners into 3d (described in calib3d.hpp)
            Mat H = findHomography( obj, scene, CV_RANSAC );
        
            //defines corners for object image with corner 0 being (0,0) and corner 2 being (col,row) -> corner1 if drawn would be square
            std::vector<Point2f> corner1(4);
            corner1[0] = cvPoint(0,0);
            corner1[1] = cvPoint( img.cols, 0 );
            corner1[2] = cvPoint( img.cols, img.rows );
            corner1[3] = cvPoint( 0, img.rows );
        
        
            std::vector<Point2f> corner2(4);
            //transforms corners calculated by findHomography function above and saves it in corner2 -> corner2 if drawn would be 2d square on 3d space
            perspectiveTransform( corner1, corner2, H);
            
            //calculate vector from trasformed points
            Point2f v1 = corner2[3] - corner2[0];
            Point2f v2 = corner2[0] - corner2[1];
            Point2f v3 = corner2[1] - corner2[2];
            Point2f v4 = corner2[2] - corner2[3];
            
            //calculate angle and length of vector using vector algebra
            double a1 = acos( (v1.x*(-v2.x)+v1.y+(-v2.y)  ) / (   sqrt(v1.x*v1.x+v1.y*v1.y)*sqrt(v2.x*v2.x+v2.y*v2.y) )   )*180/3.141592;
            double a2 = acos( (v2.x*(-v3.x)+v2.y+(-v3.y)  ) / (   sqrt(v2.x*v2.x+v2.y*v2.y)*sqrt(v3.x*v3.x+v3.y*v3.y) )   )*180/3.141592;
            double a3 = acos( (v3.x*(-v4.x)+v3.y+(-v4.y)  ) / (   sqrt(v3.x*v3.x+v3.y*v3.y)*sqrt(v4.x*v4.x+v4.y*v4.y) )   )*180/3.141592;
            double a4 = acos( (v4.x*(-v1.x)+v4.y+(-v1.y)  ) / (   sqrt(v4.x*v4.x+v4.y*v4.y)*sqrt(v1.x*v1.x+v1.y*v1.y) )   )*180/3.141592;
            
            double l1 = sqrt(v1.x*v1.x+v1.y*v1.y);
            double l2 = sqrt(v2.x*v2.x+v2.y*v2.y);
            double l3 = sqrt(v3.x*v3.x+v3.y*v3.y);
            double l4 = sqrt(v4.x*v4.x+v4.y*v4.y);
            
            //if this is initial value, save it
            if(angle1 ==999)
            {
                angle1 = a1;
                angle2 = a2;
                angle3 = a3;
                angle4 = a4;
            }
            if(length1==0)
            {
                length1 = l1;
                length2 = l2;
                length3 = l3;
                length4 = l4;
            }
            
            //if change of angle and length is within the boundary, count the object as detected
            if(
               abs(angle1 - a1) < angle1*0.2 &&
               abs(angle2 - a2) < angle2*0.2 &&
               abs(angle3 - a3) < angle3*0.2 &&
               abs(angle4 - a4) < angle4*0.2 &&
               
               abs(length1 - l1) < length1*0.2 &&
               abs(length2 - l2) < length2*0.2 &&
               abs(length3 - l3) < length3*0.2 &&
               abs(length4 - l4) < length4*0.2
               )
            {
                line( video, corner2[0], corner2[1], Scalar(0, 255, 0), 4 );
                line( video, corner2[1], corner2[2], Scalar( 0, 255, 0), 4 );
                line( video, corner2[2], corner2[3], Scalar( 0, 255, 0), 4 );
                line( video, corner2[3], corner2[0], Scalar( 0, 255, 0), 4 );
                angle1 = a1;
                angle2 = a2;
                angle3 = a3;
                angle4 = a4;
                
                length1 = l1;
                length2 = l2;
                length3 = l3;
                length4 = l4;
            
                dframe++;
            }
            else
            {
                ndframe++;
                if(ndframe%5==0)
                {
                    angle1 = 999;
                    length1 = 0;
                }
            }
        }
        else
        {
            ndframe++;
            if(ndframe%5==0)
            {
                angle1 = 999;
                length1 = 0;
            }
        }
        
        
        
        
        
        //draw final image with all variables and keypoints
       // Mat final;
       // drawMatches(img, KP1, video, KP2, good, final, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS ););
        imshow("OPENCV TEST", video);
        
        //add frame
        frame++;
        
        //exit if pressed key
        if(waitKey(30) >=0)
            break;
    }
    
    // write time record of images
    double seconds = timer.elapsedTimeSeconds();
    double percent =(dframe);
    percent = (percent/frame)*100;
    
    cout<<"-----------------------------"<<endl;
    cout<<"Total Frame in movie : "<<frame<<" frames"<<endl;
    cout<<"Total Detected Frame : "<<dframe<<" frames"<<endl;
    cout<<"Total Undetected     : "<<ndframe<<" frames"<<endl;
    cout<<"Detecting Percent    : "<<setprecision(2)<<percent<<"%"<<endl;
    cout<<"Total elapsed time   : "<<(int)seconds/60<<":";
    cout<<setfill('0')<<setw(2)<<(int)seconds%60<<endl;
    cout<<"Frame per second     : "<<(frame/seconds)<<endl;
    cout<<"Original FPS of video: "<<30<<endl;
    cout<<"-----------------------------"<<endl;
    
    ofstream mfile;
    string filename;
    string intro;
    switch(f_choose)
    {
        case 1:
        {
            filename = "../results/SIFT_";
            intro = "Result for SIFT as feature detector ";
            break;
        }
        case 2:
        {
            filename = "../results/SURF_";
            intro = "Result for SURF as feature detector ";
            break;
        }
        case 3:
        {
            filename = "../results/ORB_";
            intro = "Result for ORB as feature detector ";
            break;
        }
        case 4:
        {
            filename = "../results/BRISK_";
            intro = "Result for BRISK as feature detector ";
            break;
        }
        case 5:
        {
            filename = "../results/FAST_";
            intro = "Result for FAST as feature detector ";
            break;
        }
    }
    switch(d_choose)
    {
        case 1:
        {
            filename += "SIFT_RESULT.txt";
            intro += "and SIFT as descriptor extractor";
            break;
        }
        case 2:
        {
            filename += "SURF_RESULT.txt";
            intro += "and SURF as descriptor extractor";
            break;
        }
        case 3:
        {
            filename += "ORB_RESULT.txt";
            intro += "and ORB as descriptor extractor";
            break;
        }
        case 4:
        {
            filename += "BRISK_RESULT.txt";
            intro += "and BRISK as descriptor extractor";
            break;
        }
        case 5:
        {
            filename += "FREAK_RESULT.txt";
            intro += "and FREAK as descriptor extractor";
            break;
        }
    }
    mfile.open(filename);
    mfile<<intro<<endl;
    mfile<<endl;
    mfile<<"-----------------------------"<<endl;
    mfile<<"Total Frame in movie : "<<frame<<" frames"<<endl;
    mfile<<"Total Detected Frame : "<<dframe<<" frames"<<endl;
    mfile<<"Total Undetected     : "<<ndframe<<" frames"<<endl;
    mfile<<"Detecting Percent    : "<<setprecision(2)<<percent<<"%"<<endl;
    mfile<<"Total elapsed time   : "<<(int)seconds/60<<":";
    mfile<<setfill('0')<<setw(2)<<(int)seconds%60<<endl;
    mfile<<"Frame per second     : "<<(frame/seconds)<<endl;
    mfile<<"Original FPS of video: "<<30<<endl;
    mfile<<"-----------------------------"<<endl;
    
    
    return;
    
}

void test_ev(char* imgFile, char* videoFile)
{
    //declare variables
    char warn;
    cout<<endl;
    cout<<"Testing everything possible "<<endl;
    
    cout<<endl;
    cout<<"WARNING: this process may take very long time!"<<endl;
    
    while(warn != 'n' && warn != 'y')
    {
        cout<<"Do you wish to continue?(y/n)";
        cin>>warn;
        if(warn == 'n')
        {
            return;
        }
        else if(warn !='y')
        {
            cout<<"wrong input"<<endl;
        }
    }
    
    //read image/movie from file
    Mat img = imread(imgFile, CV_LOAD_IMAGE_GRAYSCALE);
    VideoCapture cap(videoFile);
    //VideoCapture cap(0);
    Mat video;
    
    //initialize choosing variables
    int f_choose = 0;
    int d_choose = 0;
    
    //set frame to 0;
    int frame = 0;
    int dframe = 0;
    int ndframe = 0;
    
    //set angle and length saving variable
    double angle1=999;
    double angle2=999;
    double angle3=999;
    double angle4=999;
    
    double length1 = 0;
    double length2 = 0;
    double length3 = 0;
    double length4 = 0;
    
    for(f_choose = 1; f_choose<6; f_choose++)
    {
        for(d_choose = 1; d_choose<6; d_choose++)
        {
            Time timer;
            for(;;)
            {
                //capture frame from video
                cap>>video;
                if(video.empty())
                {
                    break;
                }
                //set default virtual function for descriptor, extractor
                Ptr<FeatureDetector> descriptor;
                Ptr<DescriptorExtractor> extractor;
                
                //set blank keypoint vector
                vector<KeyPoint> KP1;
                vector<KeyPoint> KP2;
                
                //set blank Matrix for computed img, frame
                Mat computedimg;
                Mat computedcap;
                
                //set feature detector and descriptor extractor
                switch(f_choose)
                {
                    case 1:
                    {
                        descriptor = Ptr<FeatureDetector>(new SiftFeatureDetector(500));
                        break;
                    }
                    case 2:
                    {
                        descriptor = Ptr<FeatureDetector>(new SurfFeatureDetector(300));
                        break;
                    }
                    case 3:
                    {
                        descriptor = FeatureDetector::create("ORB");
                        break;
                    }
                    case 4:
                    {
                        descriptor = FeatureDetector::create("BRISK");
                        break;
                    }
                    case 5:
                    {
                        descriptor = FeatureDetector::create("FAST");
                        break;
                    }
                    case 6:
                    {
                        return;
                        break;
                    }
                }
                switch(d_choose)
                {
                    case 1:
                    {
                        extractor = DescriptorExtractor::create("SIFT");
                        break;
                    }
                    case 2:
                    {
                        extractor = DescriptorExtractor::create("SURF");
                        break;
                    }
                    case 3:
                    {
                        extractor = DescriptorExtractor::create("ORB");
                        break;
                    }
                    case 4:
                    {
                        extractor = DescriptorExtractor::create("BRISK");
                        break;
                    }
                    case 5:
                    {
                        extractor = DescriptorExtractor::create("FREAK");
                        break;
                    }
                    case 6:
                    {
                        return;
                        break;
                    }
                }
                
                //get keypoint from descriptor
                descriptor->detect(img, KP1);
                descriptor->detect(video, KP2);
                
                //compute keypoint from extractor
                extractor->compute(img, KP1, computedimg);
                extractor->compute(video, KP2, computedcap);
                
                
                //match following points
                //FlannBasedMatcher Matcher;
                BFMatcher Matcher(NORM_L2);
                vector<DMatch> matches;
                
                Matcher.match(computedimg, computedcap, matches);
                
                //calculate max and minimum value of surf/sift match's distance
                double max= 0; double min = 10000;
                
                for( int i = 0; i < computedimg.rows; i++ )
                {
                    double dist = matches[i].distance;
                    if( dist < min ) min = dist;
                    if( dist > max ) max = dist;
                }
                //for values in between minimum and maximum, save it in vector 'good'
                std::vector< DMatch > good;
                
                for( int i = 0; i < computedimg.rows; i++ )
                {
                    if( matches[i].distance <= 3*min+5 )
                    {
                        good.push_back( matches[i]);
                    }
                }
                
                //write all image with object image and webcam to final image
                Mat final;
                drawMatches(img, KP1, video, KP2, good, final, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
                //cout<<min<<"   "<<good.size()<<endl;
                
                if(good.size()>4 )
                {
                    
                    
                    //calculate corner points of image and draw a line on the final image
                    std::vector<Point2f> obj;
                    std::vector<Point2f> scene;
                    
                    for( int i = 0; i < good.size(); i++ )
                    {
                        obj.push_back( KP1[ good[i].queryIdx ].pt );
                        scene.push_back( KP2[ good[i].trainIdx ].pt );
                    }
                    // calculate the corner regarding its shift
                    // automorphs 4 corners into 3d (described in calib3d.hpp)
                    Mat H = findHomography( obj, scene, CV_RANSAC );
                    
                    //defines corners for object image with corner 0 being (0,0) and corner 2 being (col,row) -> corner1 if drawn would be square
                    std::vector<Point2f> corner1(4);
                    corner1[0] = cvPoint(0,0);
                    corner1[1] = cvPoint( img.cols, 0 );
                    corner1[2] = cvPoint( img.cols, img.rows );
                    corner1[3] = cvPoint( 0, img.rows );
                    
                    
                    std::vector<Point2f> corner2(4);
                    //transforms corners calculated by findHomography function above and saves it in corner2 -> corner2 if drawn would be 2d square on 3d space
                    perspectiveTransform( corner1, corner2, H);
                    
                    //calculate vector from trasformed points
                    Point2f v1 = corner2[3] - corner2[0];
                    Point2f v2 = corner2[0] - corner2[1];
                    Point2f v3 = corner2[1] - corner2[2];
                    Point2f v4 = corner2[2] - corner2[3];
                    
                    //calculate angle and length of vector using vector algebra
                    double a1 = acos( (v1.x*(-v2.x)+v1.y+(-v2.y)  ) / (   sqrt(v1.x*v1.x+v1.y*v1.y)*sqrt(v2.x*v2.x+v2.y*v2.y) )   )*180/3.141592;
                    double a2 = acos( (v2.x*(-v3.x)+v2.y+(-v3.y)  ) / (   sqrt(v2.x*v2.x+v2.y*v2.y)*sqrt(v3.x*v3.x+v3.y*v3.y) )   )*180/3.141592;
                    double a3 = acos( (v3.x*(-v4.x)+v3.y+(-v4.y)  ) / (   sqrt(v3.x*v3.x+v3.y*v3.y)*sqrt(v4.x*v4.x+v4.y*v4.y) )   )*180/3.141592;
                    double a4 = acos( (v4.x*(-v1.x)+v4.y+(-v1.y)  ) / (   sqrt(v4.x*v4.x+v4.y*v4.y)*sqrt(v1.x*v1.x+v1.y*v1.y) )   )*180/3.141592;
                    
                    double l1 = sqrt(v1.x*v1.x+v1.y*v1.y);
                    double l2 = sqrt(v2.x*v2.x+v2.y*v2.y);
                    double l3 = sqrt(v3.x*v3.x+v3.y*v3.y);
                    double l4 = sqrt(v4.x*v4.x+v4.y*v4.y);
                    
                    //if this is initial value, save it
                    if(angle1 ==999)
                    {
                        angle1 = a1;
                        angle2 = a2;
                        angle3 = a3;
                        angle4 = a4;
                    }
                    if(length1==0)
                    {
                        length1 = l1;
                        length2 = l2;
                        length3 = l3;
                        length4 = l4;
                    }
                    
                    //if change of angle and length is within the boundary, count the object as detected
                    if(
                       abs(angle1 - a1) < angle1*0.2 &&
                       abs(angle2 - a2) < angle2*0.2 &&
                       abs(angle3 - a3) < angle3*0.2 &&
                       abs(angle4 - a4) < angle4*0.2 &&
                       
                       abs(length1 - l1) < length1*0.2 &&
                       abs(length2 - l2) < length2*0.2 &&
                       abs(length3 - l3) < length3*0.2 &&
                       abs(length4 - l4) < length4*0.2
                       )
                    {
                        line( video, corner2[0], corner2[1], Scalar(0, 255, 0), 4 );
                        line( video, corner2[1], corner2[2], Scalar( 0, 255, 0), 4 );
                        line( video, corner2[2], corner2[3], Scalar( 0, 255, 0), 4 );
                        line( video, corner2[3], corner2[0], Scalar( 0, 255, 0), 4 );
                        angle1 = a1;
                        angle2 = a2;
                        angle3 = a3;
                        angle4 = a4;
                        
                        length1 = l1;
                        length2 = l2;
                        length3 = l3;
                        length4 = l4;
                        
                        dframe++;
                    }
                    else
                    {
                        ndframe++;
                        if(ndframe%5==0)
                        {
                            angle1 = 999;
                            length1 = 0;
                        }
                    }
                }
                else
                {
                    ndframe++;
                    if(ndframe%5==0)
                    {
                        angle1 = 999;
                        length1 = 0;
                    }
                }
                
                
                
                
                
                //draw final image with all variables and keypoints
                // Mat final;
                // drawMatches(img, KP1, video, KP2, good, final, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS ););
                //imshow("OPENCV TEST", video);
                if(frame%10==0)
                {
                    cout<<frame<<" frames passed"<<endl;
                }
                //add frame
                frame++;
                
                //exit if pressed key
            }
            
            // write time record of images
            double seconds = timer.elapsedTimeSeconds();
            double percent =(dframe);
            percent = (percent/frame)*100;
            
            cout<<"-----------------------------"<<endl;
            cout<<"Total Frame in movie : "<<frame<<" frames"<<endl;
            cout<<"Total Detected Frame : "<<dframe<<" frames"<<endl;
            cout<<"Total Undetected     : "<<ndframe<<" frames"<<endl;
            cout<<"Detecting Percent    : "<<setprecision(2)<<percent<<"%"<<endl;
            cout<<"Total elapsed time   : "<<(int)seconds/60<<":";
            cout<<setfill('0')<<setw(2)<<(int)seconds%60<<endl;
            cout<<"Frame per second     : "<<(frame/seconds)<<endl;
            cout<<"Original FPS of video: "<<30<<endl;
            cout<<"-----------------------------"<<endl;
            
            ofstream mfile;
            string filename;
            string intro;
            switch(f_choose)
            {
                case 1:
                {
                    filename = "../results/SIFT_";
                    intro = "Result for SIFT as feature detector ";
                    break;
                }
                case 2:
                {
                    filename = "../results/SURF_";
                    intro = "Result for SURF as feature detector ";
                    break;
                }
                case 3:
                {
                    filename = "../results/ORB_";
                    intro = "Result for ORB as feature detector ";
                    break;
                }
                case 4:
                {
                    filename = "../results/BRISK_";
                    intro = "Result for BRISK as feature detector ";
                    break;
                }
                case 5:
                {
                    filename = "../results/FAST_";
                    intro = "Result for FAST as feature detector ";
                    break;
                }
            }
            switch(d_choose)
            {
                case 1:
                {
                    filename += "SIFT_RESULT.txt";
                    intro += "and SIFT as descriptor extractor";
                    break;
                }
                case 2:
                {
                    filename += "SURF_RESULT.txt";
                    intro += "and SURF as descriptor extractor";
                    break;
                }
                case 3:
                {
                    filename += "ORB_RESULT.txt";
                    intro += "and ORB as descriptor extractor";
                    break;
                }
                case 4:
                {
                    filename += "BRISK_RESULT.txt";
                    intro += "and BRISK as descriptor extractor";
                    break;
                }
                case 5:
                {
                    filename += "FREAK_RESULT.txt";
                    intro += "and FREAK as descriptor extractor";
                    break;
                }
            }
            mfile.open(filename);
            mfile<<intro<<endl;
            mfile<<endl;
            mfile<<"-----------------------------"<<endl;
            mfile<<"Total Frame in movie : "<<frame<<" frames"<<endl;
            mfile<<"Total Detected Frame : "<<dframe<<" frames"<<endl;
            mfile<<"Total Undetected     : "<<ndframe<<" frames"<<endl;
            mfile<<"Detecting Percent    : "<<setprecision(2)<<percent<<"%"<<endl;
            mfile<<"Total elapsed time   : "<<(int)seconds/60<<":";
            mfile<<setfill('0')<<setw(2)<<(int)seconds%60<<endl;
            mfile<<"Frame per second     : "<<(frame/seconds)<<endl;
            mfile<<"Original FPS of video: "<<30<<endl;
            mfile<<"-----------------------------"<<endl;
            

        }
    }
        return;
    
}
