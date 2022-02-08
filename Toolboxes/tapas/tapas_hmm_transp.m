function [pvec, pstruct] = tapas_hmm_transp(r, ptrans)
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

pvec    = NaN(1,length(ptrans));
pstruct = struct;

% Number of hidden states
d = r.c_prc.n_states;

pvec(1:d-1)    = tapas_sgm(ptrans(1:d-1),1);   % ppired
pstruct.ppired = pvec(1:d-1);
pvec(d:d^2-1)  = tapas_sgm(ptrans(d:d^2-1),1); % Ared
pstruct.Ared   = pvec(d:d^2-1);

return;