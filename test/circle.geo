SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 1.0};  // Circle with radius 0.5
Curve Loop(1) = {1};
Plane Surface(1) = {1};
Physical Surface("domain", 1) = {1};
Physical Curve("boundary", 2) = {1};
Mesh.CharacteristicLengthMax = 0.05;  // Mesh size
Mesh.ElementOrder = 1;  // First-order elements
Mesh 2;
Save "circle.msh";
