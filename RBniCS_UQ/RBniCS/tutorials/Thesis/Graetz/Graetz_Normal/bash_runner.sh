#!/bin/bash
# Running River
# Giuseppe 6/9/19

#echo \*\*\*\*\*\*\*\*BETA
##for distribution in beta0505 beta205 beta052 beta7575
#for distribution in beta7575 beta0505 beta052 beta205
#do
#	echo $distribution
#	echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}.py
#	python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}.py >/dev/null
#	sed -i -e 's/HaltonBeta/GaussJacobi/' -e 's/UniformClenshawCurtisBeta/GaussJacobi/' ./sampling/distributions/beta_distribution.py >/dev/null
#	if [[ 'grep 'GaussJacobi' ./sampling/distribution/beta_distribution.py' ]]
#	then
#		echo found GaussJacobi
#	else
#		echo Error, rule not found. Exiting.
#		exit 1
#	fi
#	echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_tensor.py
#	python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_tensor.py >/dev/null
#	echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_smolyak.py
#	python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_smolyak.py >/dev/null
#	sed -i 's/GaussJacobi/UniformClenshawCurtisBeta/' ./sampling/distributions/beta_distribution.py >/dev/null
#	if [[ 'grep 'UniformClenshawCurtisBeta' ./sampling/distribution/beta_distribution.py' ]]
#	then
#		echo found UniformClenshawCurtis
#	else
#		echo Error, rule not found. Exiting.
#		exit 1
#	fi
#	#if [ $distribution == beta0505]
#	#then
#		echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_tensor.py
#		python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_tensor.py >/dev/null
#		echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_smolyak.py
#		python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_smolyak.py >/dev/null
#	#fi
#	sed -i 's/UniformClenshawCurtisBeta/HaltonBeta/' ./sampling/distributions/beta_distribution.py >/dev/null
#	if [[ 'grep 'HaltonBeta' ./sampling/distribution/beta_distribution.py' ]]
#	then
#		echo found HaltonBeta
#	else
#		echo Error, rule not found. Exiting.
#		exit 1
#	fi
#	echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton.py
#	python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton.py >/dev/null
#done
##exit 1

echo \*\*\*\*\*\*\*\*LogUniform
distribution=loguniform
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}.py >/dev/null
sed -i -e 's/HaltonLogUniform/GaussLogUniform/' -e 's/UniformClenshawCurtisLogUniform/GaussLogUniform/' -e 's/HaltonLogUniformInversion/GaussLogUniform/' ./sampling/distributions/log_uniform_distribution.py >/dev/null
if [[ 'grep 'GaussLogUniform' ./sampling/distribution/log_uniform_distribution.py' ]]
then
	echo found GaussLogUniform
else
	echo Error, rule not found. Exiting.
	exit 1
fi
#echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_tensor.py
#python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_tensor.py >/dev/null
#echo running tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_smolyak.py
#python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_smolyak.py >/dev/null
sed -i -e 's/GaussLogUniform/UniformClenshawCurtisLogUniform/' ./sampling/distributions/log_uniform_distribution.py >/dev/null
if [[ 'grep 'UniformClenshawCurtisLogUniform' ./sampling/distribution/log_uniform_distribution.py' ]]
then
	echo found UniformClenshawCurtisLogUniform
else
	echo Error, rule not found. Exiting.
	exit 1
fi
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_tensor.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_tensor.py >/dev/null
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_smolyak.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_smolyak.py >/dev/null
sed -i -e 's/UniformClenshawCurtisLogUniform/HaltonLogUniform/' ./sampling/distributions/log_uniform_distribution.py >/dev/null
if [[ 'grep 'HaltonLogUniform' ./sampling/distribution/log_uniform_distribution.py' ]]
then
	echo found HaltonLogUniform
else
	echo Error, rule not found. Exiting.
	exit 1
fi
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton.py >/dev/null
sed -i -e 's/UniformClenshawCurtisLogUniform/HaltonLogUniformInversion/' ./sampling/distributions/log_uniform_distribution.py >/dev/null
if [[ 'grep 'HaltonLogUniformInversion' ./sampling/distribution/log_uniform_distribution.py' ]]
then
	echo found HaltonLogUniformInversion
else
	echo Error, rule not found. Exiting.
	exit 1
fi
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton_inversion.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton_inversion.py >/dev/null

echo \*\*\*\*\*\*\*Uniform
distribution=uniform
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}.py >/dev/null
sed -i -e 's/HaltonUniform/GaussLegendre/' -e 's/UniformClenshawCurtisBeta/GaussLegendre/' -e 's/HaltonLogUniformInversion/GaussLegendre/' ./sampling/distributions/uniform_distribution.py >/dev/null
if [[ 'grep 'GaussLegendre' ./sampling/distribution/uniform_distribution.py' ]]
then
	echo found GaussLegendre
else
	echo Error, rule not found. Exiting.
	exit 1
fi
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_tensor.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_tensor.py >/dev/null
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_smolyak.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_gauss_smolyak.py >/dev/null
sed -i -e 's/GaussLegendre/UniformClenshawCurtisUniform/' ./sampling/distributions/uniform_distribution.py >/dev/null
if [[ 'grep 'UniformClenshawCurtisUniform' ./sampling/distribution/uniform_distribution.py' ]]
then
	echo found UniformClenshawCurtisUniform
else
	echo Error, rule not found. Exiting.
	exit 1
fi
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_tensor.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_tensor.py >/dev/null
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_smolyak.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_ucc_smolyak.py >/dev/null
sed -i -e 's/UniformClenshawCurtisUniform/HaltonUniform/' ./sampling/distributions/uniform_distribution.py >/dev/null
if [[ 'grep 'HaltonUniform' ./sampling/distribution/uniform_distribution.py' ]]
then
	echo found HaltonUniform
else
	echo Error, rule not found. Exiting.
	exit 1
fi
echo tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton.py
python3 tutorial_elliptic_ocp_weighted_poisson_${distribution}_halton.py >/dev/null
