close all; clear; clc;

import gtsam.*
import gpmp2.*
import gplfd.*

addpath(genpath('/home/afishman/Repositories/moveit_resources/panda_description'));
%%
base_pose = Pose3(Rot3(eye(3)), Point3([0,0,0]'));

%% Helper Functions
spong = @(a, d, alpha, theta) [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a * cos(theta); sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a * sin(theta); 0, sin(alpha), cos(alpha), d; 0, 0, 0, 1];

% Use the inverse of the DH matrix (from the spong function) to get the coordinates of the sphere centers

%% DH parameters

% These are parameters for the Franka arm. I had to derive them myself :(
a =       [0,     0,    0.0825, -0.0825, 0,     0.0880, 0]';
d =       [0.333, 0,    0.316,   0,      0.384, 0, 0]';
alpha_r = [-pi/2,  pi/2, pi/2,   -pi/2,  -pi/2, -pi/2, 0]';
theta = [0, 0, 0, 0, 0, 0, 0]';
abs_arm = Arm(7, a, alpha_r, d, base_pose, theta);



% physical arm
% sphere data [id x y z r]
spheres_data = [...
    % [-0.0300, 0.0000, 0.0500]'
    0     -.03      0.283  0  .10
    % [0, -0.0100, 02000]'
    0     0      0.1330 -0.01 .065
    % [0, -0.042, 0.31]'
    0     0      0.0230 -0.042 .065

    % [0, 0.05, .33]'
    1     0      0.05  -0.003 .065
    % [0.03, 0.087, .403]'
    1     0      0.03  0.087 .065
    % [0, 0.01, .49]'
    1     0      0.01  0.157 .065
    
    % [0, 0.02, .58]'
    2     -0.0825 -0.0690 -0.02 .065
    % [0.07, 0.04, .64]'
    2 -0.0125 -0.0090 -0.0400 .065
    
    % [0.07, -0.045, .64]'
    3 0.0700 -0.0450 -0.0090 .065
    % [0, 0, 0.72]'
    3 0 0 0.071 0.065
    
    % [0, 0.01, .82]'
    4 0.0 0.2130 .01 0.065
    % [0, 0.04, .93]'
    4  0    0.1030    0.0400 0.065
    % [0, 0.06, 1.02]'
    4  0    0.0130    0.0600 0.065
    
    % [0, 0, 1.02]'
    5 -0.0880    0.0000    0.0130 0.055
    % [0.06, 0, 1.03]'
    5  -0.0280    0.0000    0.0030    0.07

    % Still need to add in spheres for the last link (?)
   
];

nr_body = size(spheres_data, 1);

sphere_vec = BodySphereVector;
for i=1:nr_body
    sphere_vec.push_back(BodySphere(spheres_data(i,1), spheres_data(i,5), ...
        Point3(spheres_data(i,2:4)')));
end
arm_model = ArmModel(abs_arm, sphere_vec);


%% Plot urdf

conf = [0, 0, 0, 0, pi/4, pi/4, 0, 0, 0]';

h = figure(1); clf(1);
set(h, 'Position', [-1200, 100, 1100, 1200]);
robot = importrobot(...
    '/home/afishman/Repositories/moveit_resources/panda_description/urdf/panda.urdf', ...
    'MeshPath', ...
    {'/home/afishman/Repositories/moveit_resources/panda_description/meshes/collision'});
robot.DataFormat = 'column';

hr = show(robot, conf);
alpha(hr, 0.2);
hold on;
% 
% %% overlay gpmp2 model
hcp = [];
delete(hcp);
hcp = plotRobotModel(arm_model, conf);
plotArm(arm_model.fk_model(), conf(1:7,1), 'b', 2);
%hold off;