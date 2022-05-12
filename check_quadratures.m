function check_quadratures()
% An aux function to check and download Gauss quadratures
if (exist('lgwt', 'file')==0)
    try
        fprintf('lgwt is not found, downloading...\n');
        opts = weboptions; opts.CertificateFilename=(''); % No idea why this is needed
        websave('lgwt.zip', 'https://uk.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/4540/versions/1/download/zip', opts);
    catch ME
        error('%s. Automatic download failed. Please download lgwt from https://uk.mathworks.com/matlabcentral/fileexchange/4540-legendre-gauss-quadrature-weights-and-nodes', ME.message);
    end
    try
        unzip('lgwt.zip');
    catch ME
        error('%s. Automatic unzipping failed. Please extract lgwt.zip here', ME.message);
    end
    fprintf('Success!\n');    
end

if (exist('gauss_hermite_rule', 'file')==0)
    try
        fprintf('gauss_hermite_rule is not found, downloading...\n');
        opts = weboptions; opts.CertificateFilename=(''); % No idea why this is needed
        websave('gauss_hermite_rule.m', 'https://raw.githubusercontent.com/ezander/sglib/master/quadrature/gauss_hermite_rule.m', opts);
    catch ME
        error('%s. Automatic download failed. Please download gauss_hermite_rule.m from https://github.com/ezander/sglib', ME.message);
    end
    fprintf('Success!\n');
end

if (exist('hermite', 'file')==0)
    try
        fprintf('hermite is not found, downloading...\n');
        opts = weboptions; opts.CertificateFilename=(''); % No idea why this is needed
        websave('hermite.m', 'https://raw.githubusercontent.com/ezander/sglib/master/pce/hermite.m', opts);
    catch ME
        error('%s. Automatic download failed. Please download hermite.m from https://github.com/ezander/sglib', ME.message);
    end
    fprintf('Success!\n');
end
end
