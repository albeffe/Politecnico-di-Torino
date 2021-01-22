function [newstate, out] = shift_reg(state, con)
    % Output immediately computed
    out = state(length(state));
    new_in = mod(sum(state.*con),2);
    newstate = circshift(state,1);
    newstate(1) = new_in;
end