// Exact CPU counterpart of render.py's triangle z-buffer, for dense SAM3D meshes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
extern "C" void raster_cpu(const double* vertices, int nv, const int32_t* faces, int nf,
                           const double* K, const double* C, int H, int W,
                           const double* color, uint8_t* rgb, float* depth) {
  std::fill(rgb, rgb+H*W*3, uint8_t(245));
  std::fill(depth, depth+H*W, std::numeric_limits<float>::infinity());
  std::vector<double> camera(nv*3), uv(nv*2);
  for (int i=0;i<nv;i++) {
    for (int j=0;j<3;j++) {
      camera[3*i+j]=0;
      for (int k=0;k<3;k++) camera[3*i+j]+=(vertices[3*i+k]-C[4*k+3])*C[4*k+j];
    }
    for (int j=0;j<2;j++) {
      double p=0;
      for (int k=0;k<3;k++) p+=camera[3*i+k]*K[3*j+k];
      uv[2*i+j]=p/std::max(camera[3*i+2],1e-8);
    }
  }
  for (int f=0;f<nf;f++) {
    int a=faces[3*f],b=faces[3*f+1],c=faces[3*f+2];
    double za=camera[3*a+2],zb=camera[3*b+2],zc=camera[3*c+2];
    if(za<=0||zb<=0||zc<=0) continue;
    double ax=uv[2*a],ay=uv[2*a+1],bx=uv[2*b],by=uv[2*b+1],cx=uv[2*c],cy=uv[2*c+1];
    int x0=std::max(0,int(std::floor(std::min({ax,bx,cx}))));
    int x1=std::min(W-1,int(std::ceil(std::max({ax,bx,cx}))));
    int y0=std::max(0,int(std::floor(std::min({ay,by,cy}))));
    int y1=std::min(H-1,int(std::ceil(std::max({ay,by,cy}))));
    if(x0>x1||y0>y1) continue;
    double denominator=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy);
    if(std::abs(denominator)<1e-8) continue;
    double p[3],q[3],n[3];
    for(int j=0;j<3;j++){p[j]=camera[3*b+j]-camera[3*a+j];q[j]=camera[3*c+j]-camera[3*a+j];}
    n[0]=p[1]*q[2]-p[2]*q[1];n[1]=p[2]*q[0]-p[0]*q[2];n[2]=p[0]*q[1]-p[1]*q[0];
    double shade=.45+.55*std::abs(n[2])/std::max(std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]),1e-10);
    for(int y=y0;y<=y1;y++) for(int x=x0;x<=x1;x++) {
      double u=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/denominator;
      double v=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/denominator;
      double w=1-u-v;
      if(u<0||v<0||w<0) continue;
      double z=1/std::max(u/za+v/zb+w/zc,1e-8);
      int index=y*W+x;
      if(z<depth[index]) {
        depth[index]=float(z);
        for(int j=0;j<3;j++) rgb[3*index+j]=uint8_t(color[j]*shade);
      }
    }
  }
  for(int i=0;i<H*W;i++) if(!std::isfinite(depth[i])) depth[i]=0;
}
