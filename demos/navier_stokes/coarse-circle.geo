SetFactory("OpenCASCADE");
a() = ShapeFromFile("/Users/robert_kirby/Code/Irksome/demos/navier_stokes/circle.step");

Mesh.CharacteristicLengthMin = 0.1;
Mesh.CharacteristicLengthMax = 0.1;
                Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Line(5) = {5};
Physical Line(6) = {6};
Physical Line(7) = {7};
Physical Line(8) = {8};
Physical Line(9) = {9};
Physical Line(10) = {10};
Physical Line(11) = {11};
Physical Line(12) = {12};
Physical Line(13) = {13};
Physical Line(14) = {14};
Physical Line(15) = {15};
Physical Surface(16) = {1};
