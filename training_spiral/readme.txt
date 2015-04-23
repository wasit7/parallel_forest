FFT
รูปขนาด 1024 x 768 จำนวน 1242 รูป 
- กำหนดจำนวน dataset โดย dset=2
-ในแต่ละ dataset จะทำการ random จำนวนรูปมา ตามจำนวน dfiles=50
-ใน 1 รูปทำการ random pixel มา 10 จุด (samples) แต่ละจุดมี boundary 100 x 100 (bs) 

-ใน 1 จุด จะได้ feature vector มา 20 dimension (wd)
-class (clmax) ที่เป็นไปได้คือ 0,1,2,3,4,5,6,7,8,9,255
