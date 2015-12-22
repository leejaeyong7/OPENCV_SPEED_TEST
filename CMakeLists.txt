
#utilities test
project (Test)
cmake_minimum_required(VERSION 2.8)


#--------------------------#
#----- Find Libraries -----#
#--------------------------#

#----- OpenCV -----#
find_package( OpenCV )
#--- end OpenCV ---#




#--------------------------#
#--- end Find Libraries ---#
#--------------------------# 



#-------------------------------------#
#----- add executables and links -----#
#-------------------------------------#

#----- OpenCV Features -----#
if ( OpenCV_FOUND )
	add_executable( test opencvt.cpp timer.cpp)
	target_link_libraries( test ${OpenCV_LIBS})
endif ( OpenCV_FOUND )
#--- end OpenCV Features ---#

#-------------------------------------#
#--- end add executables and links ---#
#-------------------------------------#
