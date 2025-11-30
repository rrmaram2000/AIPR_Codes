function [imout,orderout,cimout,meta,thetaout]=scat_display_with_borders(in,dirac,options,scatt)
%SCAT_DISPLAY_WITH_BORDERS Enhanced version of scat_display with clear borders
%
% This function creates scattering disks with visible borders between
% different scale and orientation components.
%
% Based on scat_display.m from ISCV paper
% Enhanced for publication-quality figures with clear component boundaries

options.null=0;
use_whole_ener=getoptions(options,'use_whole_ener',0);
type=getoptions(options,'display_type',2);
logpolar=getoptions(options,'display_logpolar',1);
renorm_process=getoptions(options,'renorm_process',0);
border_width=getoptions(options,'border_width',2);  % NEW: Border thickness
border_color=getoptions(options,'border_color',0);  % NEW: Border color (0=black)

maxsize=getoptions(options,'display_size',512);
J=in{1}.meta.j;
L=max(in{2}.meta.theta);

%create the legacy 'meta' structure
meta = recover_meta(in, dirac);
if nargin < 4
scatt = meta.ave;
end

meta.covered=zeros(size(meta.order));
meta=effective_energy(meta,use_whole_ener);
meta=compute_rectangles(meta,use_whole_ener,type==0,logpolar);
maxorder=max(meta.order);

if renorm_process
  norm_ratio = scatt./meta.dirac_norm;
else
  l2_renorm = getoptions(options,'l2_renorm',0);
  denom=ones(size(meta.dirac_norm));
  if l2_renorm
    denom=sqrt(2.^(-mod(meta.scale,J)));
  end
  if (size(scatt)~=size(denom))
    scatt = scatt';
  end
  norm_ratio = scatt./denom;
end

heights=meta.rectangle(:,2)-meta.rectangle(:,1);
widths=meta.rectangle(:,4)-meta.rectangle(:,3);

fact_h = maxsize;
fact_w = maxsize;

if type < 2
  imout=zeros(fact_h+1,fact_w+1);
  orderout=zeros(fact_h+1,fact_w+1);
else
  for m=1:maxorder
    imout{m}=zeros(fact_h,fact_w);
    border_mask{m}=zeros(fact_h,fact_w);  % NEW: Border mask
  end
end

if type==1
  %concatenate the rectangles along orders
  for ord=1:maxorder
    selected=find(meta.order==ord);
    ordener(ord)=sum(meta.dirac_norm(selected).^2);
  end
  ordener=ordener/sum(ordener);
  cordener=cumsum(ordener);
  cordener=[0 cordener];
  lower=0;
  for ord=1:maxorder
    selected=find(meta.order==ord);
    if ordener(ord)>0
      if logpolar
        upper=sqrt(lower^2+ordener(ord));
        meta.rectangle(selected,3:4)=sqrt(meta.rectangle(selected,3:4).^2*upper^2+...
          (1-meta.rectangle(selected,3:4).^2)*lower^2);
        lower=upper;
      else
        meta.rectangle(selected,3:4)=ordener(ord)*meta.rectangle(selected,3:4)+cordener(ord);
      end
    end
  end
  type=0;
end

switch type
  case 0
    for l=1:length(norm_ratio)
      ext(1)=1+floor(fact_h*meta.rectangle(l,1));
      ext(2)=1+floor(fact_h*meta.rectangle(l,2));
      ext(3)=1+floor(fact_w*meta.rectangle(l,3));
      ext(4)=1+floor(fact_w*meta.rectangle(l,4));

      inthh=[ext(1):ext(2)];
      intww=[ext(3):ext(4)];
      imout(inthh,intww)=norm_ratio(l);
      orderout(inthh,intww)=meta.order(l);
    end
    if logpolar
      [imout,thetaout]=logpolar_conversion(imout,L);
      [orderout]=logpolar_conversion(orderout,L);
    end
    m1=(orderout==2);
    m2=(orderout==3);
    m3=(orderout>3);
    [gox,goy]=gradient(orderout);
    couronnes=(conv2(gox.^2+goy.^2,ones(5),'same') < .25);
    nimout=imout/max(imout(:));
    nimout=nimout.*couronnes+(1-couronnes);
    [NN,MM]=size(nimout);
    cimout=ones(NN,MM,3);
    cimout(:,:,1)=1-nimout.*(m2|m3);
    cimout(:,:,3)=1-nimout.*(m1|m3);
    cimout(:,:,2)=1-nimout.*(m1|m2);

  case 2
    % NEW: Fill rectangles and create border mask
    for ord=2:maxorder
      selected=find(meta.order==ord);
      for l=selected
        inth=min(fact_h,[1+floor(fact_h*meta.rectangle(l,1)):floor(fact_h*meta.rectangle(l,2))]);
        intw=min(fact_w,[1+floor(fact_w*meta.rectangle(l,3)):floor(fact_w*meta.rectangle(l,4))]);

        % Fill the rectangle with coefficient value
        imout{ord-1}(inth,intw)=norm_ratio(l);

        % NEW: Mark borders
        if border_width > 0
          % Top and bottom borders
          border_h_top = inth(1:min(border_width, length(inth)));
          border_h_bot = inth(max(1, end-border_width+1):end);

          % Left and right borders
          border_w_left = intw(1:min(border_width, length(intw)));
          border_w_right = intw(max(1, end-border_width+1):end);

          % Mark borders in mask
          border_mask{ord-1}(border_h_top, intw) = 1;
          border_mask{ord-1}(border_h_bot, intw) = 1;
          border_mask{ord-1}(inth, border_w_left) = 1;
          border_mask{ord-1}(inth, border_w_right) = 1;
        end
      end

      if logpolar
        [imout{ord-1}]=logpolar_conversion(imout{ord-1},L);
        if border_width > 0
          [border_mask{ord-1}]=logpolar_conversion(border_mask{ord-1},L);
        end
      end

      % NEW: Apply borders as dark lines
      if border_width > 0
        border_indices = border_mask{ord-1} > 0.5;
        imout{ord-1}(border_indices) = border_color * max(imout{ord-1}(:));
      end
    end
end

end

% Helper functions from original scat_display.m
function meta=compute_rectangles(meta,use_whole_ener,unified_plot, logpolar)
LP_correction=1;
R=length(meta.order);
maxorder=max(meta.order);
first_mask=find(meta.order==2);
J=max(meta.scale(first_mask))+1;
L=max(meta.orientation(first_mask))+1;

meta.lp_correction=1;
meta.rectangle(1,:)=[0 1 0 sqrt(meta.lp_correction)];
meta.dirac_onorm=meta.dirac_norm;

for o=1:maxorder-1
  slice=find(meta.order==o);
  for s=slice
    if o==1
      children=find(meta.order==o+1);
    else
      children=find((floor(meta.scale/J)==meta.scale(s))&(floor(meta.orientation/L)==meta.orientation(s))&(meta.order==o+1));
    end
    if ~isempty(children)
      [newrectangles,outrect]=split_rectangle(meta.rectangle(s,:),meta.scale(children),...
        meta.orientation(children),meta.dirac_norm(s),meta.dirac_onorm(s),...
        meta.dirac_effnorms(children),J,L,use_whole_ener,unified_plot,logpolar,o);
      for c=1:length(children)
        meta.rectangle(children(c),:)=newrectangles(c,:);
      end
      meta.covered(s) = ((outrect(2)-outrect(1))*(outrect(4)-outrect(3))+...
        sum((newrectangles(:,2)-newrectangles(:,1)).*(newrectangles(:,4)-newrectangles(:,3))))/...
        ((meta.rectangle(s,2)-meta.rectangle(s,1))*(meta.rectangle(s,4)-meta.rectangle(s,3)));
      meta.rectangle(s,:)=outrect;
    end
  end
end
end

function [out,outlowp]=split_rectangle(inrectangle, scales, orientations, dirac_phi, dirac_orig,dirac_norms,J,L,use_whole_ener,unified_plot, logpolar,order)
C=length(dirac_norms);
if unified_plot | order==1
  ener=dirac_phi^2;
  totener=sum(dirac_norms)+ener;
else
  totener=sum(dirac_norms);
end
if use_whole_ener
  totener=dirac_orig^2;
end

totalheight=inrectangle(2)-inrectangle(1);
if logpolar
  totalwidth=inrectangle(end)^2-inrectangle(end-1)^2;
else
  totalwidth=inrectangle(end)-inrectangle(end-1);
end

outlowp=inrectangle;
if unified_plot | order==1
  if logpolar
    outlowp(end)=sqrt(outlowp(end-1)^2+totalwidth*(ener/totener));
  else
    outlowp(end)=outlowp(end-1)+totalwidth*(ener/totener);
  end
  rasterwidth=outlowp(end);
else
  rasterwidth=inrectangle(3);
end
scale_parent= mod(floor(scales/J),J);
orient_parent=mod(floor(orientations/L),L);
diffori=1;

for j=J-1:-1:0
  pack=find(mod(scales,J)==j);
  if ~isempty(pack)
    width=sum(dirac_norms(pack).^1);
    rasterheight=inrectangle(1);
    for l=0:L-1
      ind=find(mod(orientations(pack)-diffori*orient_parent(pack)+(order>1)*L/2,L)==l);
      out(pack(ind),1)=rasterheight;
      out(pack(ind),2)=rasterheight+totalheight*dirac_norms(pack(ind)).^1/width;
      out(pack(ind),3)=rasterwidth;
      if logpolar
        out(pack(ind),4)=sqrt(rasterwidth^2+totalwidth*width/totener);
      else
        out(pack(ind),4)=rasterwidth+totalwidth*width/totener;
      end
      rasterheight=out(pack(ind),2);
    end
    if logpolar
      rasterwidth=sqrt(rasterwidth^2+totalwidth*width/totener);
    else
      rasterwidth=rasterwidth+totalwidth*width/totener;
    end
  end
end
end

function metaout=effective_energy(meta,use_whole_ener)
R=length(meta.order);
maxorder=max(meta.order);
first_mask=find(meta.order==2);
J=max(meta.scale(first_mask))+1;
L=max(meta.orientation(first_mask))+1;

last_mask=find(meta.order==maxorder);
metaout=meta;
metaout.dirac_effnorms(last_mask)=meta.dirac_norm(last_mask).^2;
if use_whole_ener
  metaout.dirac_effnorms(last_mask)=meta.dirac_onorm(last_mask).^2;
end

for o=maxorder-1:-1:1
  slice=find(meta.order==o);
  for s=slice
    if o==1
      children=find(meta.order==o+1);
    else
      children=find((floor(meta.scale/J)==meta.scale(s))&(floor(meta.orientation/L)==meta.orientation(s))&(meta.order==o+1));
    end
    metaout.dirac_effnorms(s)=sum(metaout.dirac_effnorms(children))+meta.dirac_norm(s).^2;
    if use_whole_ener
      metaout.dirac_effnorms(s)=metaout.dirac_onorm(s).^2;
    end
  end
end
end

function [out,theta]=logpolar_conversion(in,L)
[N,M]=size(in);

ix=-M:M;
iix=ones(length(ix),1)*ix;
iiy=iix';
r=sqrt(iix.^2+iiy.^2);
theta=angle(iiy+i*iix);

[N2,M2]=size(r);
mask=(r>M);
theta=mod(theta+(0*L+1)*pi/(2*L),pi);
theta=theta/pi;

code=min(N-2,max(1,round(theta*N)))+N*min(M-1,max(1,round(r)));
out=in(max(1,min(numel(in),1+code)));
out(mask)=0;
clear theta;
theta(:,:,1)=mod(code+1,N);
theta(:,:,2)=min(M-1,max(1,round(r)));
end
