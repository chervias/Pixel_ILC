#include <healpix_cxx/healpix_base.h>
#include <healpix_cxx/rangeset.h>
#include <healpix_cxx/pointing.h>
#include <healpix_cxx/vec3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
using namespace std;

extern "C" {
	void query_disc_wrapper(long ipix, double radius, int nside, long* ipix_arr, int* nipix, int *sucess){
		// first, we need to transform ipix to a pointing center
		// We define the hp_base. I don't know if it is faster in nested, CHECK THIS !!!
		T_Healpix_Base<long> hp_base(nside,RING,SET_NSIDE);
		int ipixx = (int) ipix; 
		pointing center = hp_base.pix2ang(ipixx);
		vec3 center_v = center.to_vec3();
		rangeset<long> pp;
		try{
			hp_base.query_disc(center,radius,pp);
			std::vector<long> v = pp.toVector();
			*nipix	= (int) v.size();
			for(std::size_t i = 0; i < v.size(); i++) {
				ipix_arr[i]	= (long) v[i];
				// Here for the calculation of the angular distance, I use the same recipe as in healpy
				//pointing center_pix = hp_base.pix2ang(v[i]);
				//vec3 center_pix_v = center_pix.to_vec3();
				//vec3 cross = crossprod(center_v,center_pix_v);
				//double vecprod = cross.Length() ;
				//double scalprod = dotprod(center_v,center_pix_v);
				//pixel_distances[i] = atan2(vecprod , scalprod ) ;
			}
			*sucess = 1;
		}
		catch (PlanckError e){
			*sucess = 0;
		}
	}
}