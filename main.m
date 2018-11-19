 im  = readNPY('img.npy'); 
 img1 = readNPY('img01py.npy');
 img2 = readNPY('img10py.npy');
 img3 = readNPY('img0_1py.npy');
 img4 = readNPY('img_10py.npy');

 x = im(:);
 x1 = img1(:);
 x2 = img2(:);
 x3 = img3(:);
 x4 = img4(:);

 count = nnz(x);
 display(count);
 G_mat = randn(5*count,2401);
 A = orth(G_mat')';
 
 y = A*x;
 y1 = A*x1;
 y2 = A*x2;
 y3 = A*x3;
 y4 = A*x4;

 x0 = A'*y;
 x01 = A'*y1;
 x02 = A'*y2;
 x03 = A'*y3;
 x04 = A'*y4;
 
 tic
 xp = l1eq_pd(x0, A, [], y, 1e-3);
 toc
 
 tic
 xp1 = l1eq_pd(x01, A, [], y, 1e-3);
 toc
 
 tic
 xp2 = l1eq_pd(x02, A, [], y, 1e-3);
 toc
 
 tic
 xp3 = l1eq_pd(x03, A, [], y, 1e-3);
 toc
 
 tic
 xp4 = l1eq_pd(x04, A, [], y, 1e-3);
 toc
 
 xf = (xp1 .+ xp2 .+ xp3 .+ xp4)/4;
 
 img = reshape(x, 49, 49);
 figure
 imshow(img);
 title('Target Image');

 img = reshape(x1, 49, 49);
 figure
 imshow(img);
 title('1st Input Image');
 
 img = reshape(xp1, 49, 49);
 figure
 imshow(img);
 title('1st Reconstructed Image');
 
 img = reshape(xf, 49, 49);
 figure
 imshow(img);
 title('Reconstructed Image');
 