
// Concave polygon (triangular meshable). Points and lines marked on the boundary.

SetFactory("OpenCASCADE"); // robust boolean/loop handling

// --- Points (x, y, z, mesh size)
Point(1) = {0.0,  0.0, 0, 0.12};
Point(2) = {3.0,  0.0, 0, 0.12};
Point(3) = {3.0,  1.0, 0, 0.12};
Point(4) = {2.0,  0.5, 0, 0.06}; // inward notch -> makes shape concave
Point(5) = {3.0,  2.0, 0, 0.12};
Point(6) = {0.0,  2.0, 0, 0.12};
Point(7) = {-0.5, 1.0, 0, 0.12};

// --- Lines (boundary segments)
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 1};

// --- Curve loop and surface
Curve Loop(1) = {1,2,3,4,5,6,7};
Plane Surface(1) = {1};

// --- Physical tags for importing downstream
// mark all boundary points explicitly (you asked points marked "boundary")
Physical Point("boundary") = {1,2,3,4,5,6,7};

// also mark boundary lines and the surface
Physical Line("boundary_lines") = {1,2,3,4,5,6,7};
Physical Surface("domain") = {1};

// optional: force triangular 2D mesh
Mesh.Algorithm = 6; // Frontal Delaunay for triangles (nice triangles)
Mesh 2;
