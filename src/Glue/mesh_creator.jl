export create_circle_mesh

function create_circle_mesh(mesh_circle_size::Float64=0.05, order::Int=1)
    return """
    SetFactory("OpenCASCADE");
    Circle(1) = {0, 0, 0, 1.0};  // Circle with radius 1.0
    Curve Loop(1) = {1};
    Plane Surface(1) = {1};
    Physical Surface("domain", 1) = {1};
    Physical Curve("boundary", 2) = {1};
    Mesh.CharacteristicLengthMax = $mesh_circle_size;  // Mesh size
    Mesh.ElementOrder = $order;  // Element order
    Mesh 2;
    Save "circle.msh";
    """
end
