% Stupid low rank matrix class for fast matvecs
classdef lrmatrix
    properties( SetAccess = public, GetAccess = public )
        Lfactor;
        Rfactor;
    end
    
    methods (Access = public)
        % Constructor
        function lrm = lrmatrix(newLfactor, newRfactor)
            assert(size(newLfactor,2)==size(newRfactor,2), 'L and R factors must be n x r and m x r matrices resp.');
            lrm.Lfactor = newLfactor;
            lrm.Rfactor = newRfactor;
        end
        
        % All Size options
        function [n,m] = size(lrm,varargin)
            n = size(lrm.Lfactor,1);
            m = size(lrm.Rfactor,1);
            nm = [n m];
            if (nargout<2)
                n = nm;
            end
            if (numel(varargin)==1)
                n = nm(varargin{1});
            end
        end
        function [lrm] = reshape(lrm,varargin)
            % stub for amen
        end
        % Always return 1 for issparse to prevent rank grumbling in amen
        function [yes] = issparse(varargin)
            yes = true;
        end
        
        % Most important: multiplications
        function [y] = mtimes(a,b)
            if (isa(a, 'lrmatrix'))
                y = a.Rfactor.'*b;
                if (isscalar(b))
                    y = lrmatrix(a.Lfactor, y.');
                else
                    y = a.Lfactor*y;
                end
            elseif (isa(b, 'lrmatrix'))
                y = a*b.Lfactor;
                if (isscalar(a))
                    y = lrmatrix(y, b.Rfactor);
                else
                    y = y*b.Rfactor.';
                end
            else 
                error('WTF, LRmatrix mtimes called for non-LRmatrix arguments');
            end
        end
        
        % Summations
        function [y] = plus(a,b)
            if (isa(a, 'lrmatrix') && isa(b,'lrmatrix'))
                y = lrmatrix([a.Lfactor, b.Lfactor], [a.Rfactor, b.Rfactor]);
            elseif (isa(a, 'lrmatrix'))
                % Summation with full goes into full
                if (issparse(b))
                    y = sparse(a.Lfactor)*sparse(a.Rfactor.');
                else
                    y = a.Lfactor*a.Rfactor.';
                end
                y = y+b;
            elseif (isa(b, 'lrmatrix'))
                if (issparse(a))
                    y = sparse(b.Lfactor)*sparse(b.Rfactor.');
                else
                    y = b.Lfactor*b.Rfactor.';
                end
                y = a+y;
            else 
                error('WTF, LRmatrix plus called for non-LRmatrix arguments');
            end
        end
        
        % Transpositions
        function [lrm] = ctranspose(lrm)
            tmp = lrm.Lfactor;
            lrm.Lfactor = conj(lrm.Rfactor);
            lrm.Rfactor = conj(tmp);
        end
        function [lrm] = transpose(lrm)
            tmp = lrm.Lfactor;
            lrm.Lfactor = lrm.Rfactor;
            lrm.Rfactor = tmp;
        end        
    end
end
