%{
 Simulation code for for Algorithm 1 presented in the following scientific paper:
 "Fast Converging Algorithm for Weighted Sum Rate Maximization in Multicell 
 MISO Downlink by Le-Nam Tran, Muhammad Fainan Hanif,Antti Tolli, Markku Juntti,
 IEEE Signal Processing Letters 19.12 (2012): 872-875
%}
clear variables
clc

% Number of RX antennas
M=1;

% Number of antennas for each BS
nTx=8;

% total users
nUsers=4;

% No. of cells
nCells=2;

% for nUsers=4, usercellindex = [1 1 2 2], the first two elements give the users served
% by BS 1 and the other two elements give indeces of users served by BS 2
usercellindex=[1 1 2 2];

noise_power = 1; % noise power
powerpercell = 12; % maximum tx power per cell in dB (i.e., normalized with noise power)

powerpercell = 10.^(powerpercell./10)*ones(nCells,1); 

nIterations = 200;% maximum # of iterations

weights = [1; 1; 1; 1]; % for sum rate maximization

% generate channels
channel = sqrt(1/2)*(randn(M, nTx, nUsers, nCells)+ ...
                    1i*randn(M, nTx, nUsers, nCells));

%% Generate a feasible initial point
% Procedure: first generate  beamformers that stisfy the power constraints 
% and calculate \phi 
winit = rand(nTx,nUsers) + 1i*rand(nTx,nUsers); % randn can also be used here
for iCell = 1:nCells
    winit(:,usercellindex==iCell) = sqrt(powerpercell(iCell))/norm(winit(:,usercellindex==iCell))*...
        winit(:,usercellindex==iCell); % scalling to meet the power constraint
end
mybetaInit = zeros(nUsers,1);
xInit = zeros(nUsers,1);
for iUser=1:nUsers
    % find the index of the other users
    otherusers = find(1:nUsers ~= iUser);
    mybetaInit(iUser) = norm([noise_power;diag(((reshape(channel(:,:,iUser,...
        usercellindex(otherusers)),nTx,[])).')*(winit(:,otherusers)))]); % set (6d) to equality
    xInit(iUser) = (abs(channel(:,:,iUser,usercellindex(iUser))*winit(:,iUser))/mybetaInit(iUser))^2;
    
end
phi = sqrt(xInit)./mybetaInit;
tNext = (1+xInit).^weights;

%% To generate Fig. 2, uncomment the following four lines
% load('channel.mat')
% tNext=ones(nUsers,1);        
% phi = 1./[0.142857142857143;0.214285714285714;0.285714285714286;0.357142857142857];
% weights = [0.14;0.21;0.28;0.36]; % weights in Fig. 2

% scale the weights so that all are larger than 1 
scalecoeff = 1.1/min(weights);
weights = scalecoeff*weights;


% error tolerance. Algorithm terminates if the increase between
% two iterations < tol
tol = 1e-3;   

% memory allocation
sumrate = zeros(nIterations,1); % the sequence of sum rate
weightedsumrate = zeros(nIterations,1); % the sequence of weighted sum rate
seqobj = zeros(nIterations,1); % the sequence of the objective of subproblems

% define optimization variables
t     = sdpvar(nUsers,1);
mybeta   = sdpvar(nUsers,1);
x     = sdpvar(nUsers,1);
w     = sdpvar(nTx,nUsers,'full','complex'); % beamformers
%% main loop
ops =  sdpsettings('solver','mosek','verbose',0); % set the interal solver to be MOSEK
for iIteration=1:nIterations
    
    constraints = []; % contain all the constraints
    obj = -geomean(t); % yalmip automatically implements (11b) and (11c)
    
    for iUser=1:nUsers
        b = (channel(:,:,iUser,usercellindex(iUser))*w(:,iUser)) - 1/(phi(iUser)*2)*(x(iUser));
        
        % constraint (11d) in the paper
        constraints = [constraints,cone([0.5*(b-1);sqrt(phi(iUser)/2)*(mybeta(iUser))],0.5*(b+1))] ;
        
        % constraint (11e)
        constraints = [constraints, x(iUser) >= tNext(iUser)^(1/weights(iUser))-1 ...
            + 1/weights(iUser)*(tNext(iUser)^(1/weights(iUser)-1))*(t(iUser)-tNext(iUser))];
        
        % find the index of the other users
        otherusers = find(1:nUsers ~= iUser);
        
        interference = [noise_power;diag(((reshape(channel(:,:,iUser,usercellindex(otherusers)),nTx,[])).')*(w(:,otherusers)))];
        
        %constraint (11f)
        constraints = [constraints,cone(interference,mybeta(iUser))]; %constraint (11f)
        
    end
    
    % implicit constraints in t (refer to (6b)) and z (hyperbolic constraints)
    constraints=[constraints,t>=1];
    
    % constraint (11g)
    for iCell=1:nCells
        constraints=[constraints,cone(vec(w(:,usercellindex==iCell)),sqrt(powerpercell(iCell)))];
    end
    diagnostics = solvesdp(constraints,obj,ops); % solve the problem
    if (diagnostics.problem==0) % sucessfully solve
        % Step 3 of Algorithm 1 in the paper
        phi = double(x.^(0.5)./mybeta);
        tNext=double(t);
        
        %compute objective sequence returned by the solver after each iteration
        seqobj(iIteration) = sum(log2(double(t)));
        
        % save the current values of beamformers
        beamformer = double(w);
        
        % compute the WSR and SR for each iteration
        for iUser=1:nUsers
            otherusers = find(1:nUsers ~= iUser);
            interference = [noise_power;diag(((reshape(channel(:,:,iUser,...
                usercellindex(otherusers)),nTx,[])).')*((beamformer(:,otherusers))))];
            
            % compute SINR for each user
            SINR=abs(channel(:,:,iUser,usercellindex(iUser))*beamformer(:,iUser))^2/((norm(interference))^2);
            
            % WSR and SR calculation
            weightedsumrate(iIteration) = weightedsumrate(iIteration) + weights(iUser)*log2(1+SINR);
            sumrate(iIteration)  = sumrate(iIteration)+log2(1+SINR);
        end
        % check stopping criterion
        if (iIteration>1)&&(abs(weightedsumrate(iIteration)-weightedsumrate(iIteration-1)) < tol)
            seqobj(min(iIteration+1,nIterations):end)=[];
            sumrate(min(iIteration+1,nIterations):end)=[];
            weightedsumrate(min(iIteration+1,nIterations):end)=[];
            break; %  converge, break
        end
        
    else % numerical issue
        disp('There may be a numerical issue in the solver. Early termination');
        break; 
    end
end
seqobj = seqobj/scalecoeff;
weightedsumrate = weightedsumrate/scalecoeff;
sumrate = sumrate/scalecoeff;

plot(1:length(weightedsumrate),weightedsumrate,'-rs','MarkerEdgeColor','r',...
                'MarkerFaceColor','r');
hold on
plot(1:length(seqobj),seqobj,'-bd','MarkerEdgeColor','b',...
                'MarkerFaceColor','b');
grid on            
xlim([1-0.1 length(seqobj)]);

xlabel('Iteration index');
ylabel('Weighted sum rate (b/s/Hz)');
h=legend( 'Weighted sum rate','The objective sequence of convex approximate subproblems','Location','SouthEast');
set(h,'Color', 'w','box','on','EdgeColor','w');
saveas(gcf, '../../results/ConvergencePlot.png')