#include <boost/math/special_functions/gamma.hpp>

namespace chi_squared {

double Cdf(double x, double dof) { return boost::math::gamma_p(0.5 * dof, x); }
} // namespace chi_squared
