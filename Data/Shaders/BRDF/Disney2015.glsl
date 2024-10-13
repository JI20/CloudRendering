/**
 * MIT License
 *
 * Copyright (c) 2024, Christoph Neuhauser, Jonas Itt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define BRDF_SUPPORTS_SPECULAR
#define BRDF_SUPPORTS_TRANSMISSION
#define BRDF_SUPPORTS_SUBSURFACE_SCATTERING

// --------- BRDF PRIVATE Functions --------------

// 0. Helper Functions

float atan2(in float y, in float x) {
    bool s = (abs(x) > abs(y));
    return mix(PI / 2.0 - atan(x,y), atan(y,x), s);
}

float avoidZero(float x, float y)
{
    if ((abs(x) > abs(y)))
    {
        return x;
    }
    else
    {
        return x > 0 ? y : -y;
    }
}

// Source: PBR Book v3: https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
float fresnelDielectric(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    // Potentially swap indices of refraction
    // TODO: See if swap is correct
    bool entering = cosThetaI > 0.0;
    if (!entering)
    {
        // Swap eta values
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
        cosThetaI = abs(cosThetaI);
    }

    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = sqrt(max(0.0, 1.0 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 1.0)
        return 1.0;
    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0;
}

// 0.1 Sheen Lobe
// Source: https://schuttejoe.github.io/post/disneybsdf/
vec3 calculateTint(vec3 baseColor) {
	float lum = dot(vec3(0.3, 0.6, 1.0), baseColor);
	return lum > 0.0 ? baseColor * (1.0/lum) : vec3(1.0);
}

// Source: https://schuttejoe.github.io/post/disneybsdf/
vec3 evaluateSheen(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 baseColor) {
	if(parameters.sheen <= 0.0) {
		return vec3(0.0);
	}

	float LdotH = dot(lightVector, halfwayVector);
	vec3 tint = calculateTint(baseColor);
	float f = (1.0 - LdotH) * (1.0 - LdotH);
	float fresnelSchlickWeight = f * f * (1.0-LdotH);
	return parameters.sheen * mix(vec3(1.0), tint, parameters.sheenTint) * fresnelSchlickWeight;
}

// 0.2 Clearcoat Lobe
// Source: https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L49C1-L55C2
float GTR1(float NdotH, float a) {
    if (a >= 1.0) {
		return 1.0 / M_PI;
	}
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NdotH * NdotH;
    return (a2 - 1.0) / (PI * log(a2) * t);
}

// Source: https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L69C1-L74C2
float smithG_GGX(float NdotV, float alphaG) {
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;

    return 1.0 / (max(NdotV,-0.99) + sqrt(a + b - a * b));;
}

float evaluateClearcoat(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector, out float samplingPDF, flags hitFlags)
{
	if(parameters.clearcoat <= 0.0) {
		return 0.0;
	}

	float NdotH = abs(dot(normalVector, halfwayVector));
	float NdotL = abs(dot(normalVector, lightVector));
	float NdotV = abs(dot(normalVector, viewVector));
	float LdotH = dot(lightVector, halfwayVector);
    float VdotH = dot(viewVector, halfwayVector);
	
    float sinThetaH = sqrt(1.0 - (min(NdotH * NdotH, 0.95)));

	float fsw0 = (1.0 - LdotH) * (1.0 - LdotH);
	float fresnelSchlickWeight = fsw0 * fsw0 * (1.0-LdotH);

	float d = GTR1(NdotH, mix(0.1, 0.001, parameters.clearcoatGloss));
	float f = mix(0.04,1.0,fresnelSchlickWeight);
	float gl = smithG_GGX(NdotL, 0.25);
	float gv = smithG_GGX(NdotV, 0.25);
    
    
    if (hitFlags.clearcoatHit) {
        samplingPDF = d * NdotH * sinThetaH;
        return parameters.clearcoat * f * gl * gv * NdotL * VdotH / NdotH;
    }
    
	// TODO: * 0.25 right?
    return parameters.clearcoat * d * f * gl * gv * NdotL * VdotH * sinThetaH;
}

// 0.3 Specular lobe (BRDF)
// Source: https://schuttejoe.github.io/post/disneybsdf/
float anisoGgxD(vec3 halfwayVector, vec3 normalVector, float ax, float ay)
{
    float hX2 = halfwayVector.x * halfwayVector.x;
    float hY2 = halfwayVector.z * halfwayVector.z;
    float NdotH2 = dot(halfwayVector, normalVector);
	NdotH2 *= NdotH2;
    float ax2 = ax*ax;
    float ay2 = ay*ay;

    return 1.0 / (M_PI * ax * ay * sqr(hX2 / ax2 + hY2 / ay2 + NdotH2));
}

// https://jcgt.org/published/0003/02/03/, page 86
// https://cseweb.ucsd.edu/~tzli/cse272/wi2023/homework1.pdf
float anisoHeitzGgxG(vec3 vec, vec3 halfwayVector, float ax, float ay)
{
    float dotH = dot(vec, halfwayVector);
    if (dotH <= 0.0) {
        return 0.0;
    }

	// Todo: Find out if vector.x is correct or if it should be lightVector.x
    float lambda = 0.5 * (-1.0 + sqrt(1.0 + (vec.x*vec.x*ax*ax + vec.y*vec.y*ay*ay)/(vec.z*vec.z)));
    return 1.0 / (1.0 + lambda);
}

float schlickR0FromRelativeIOR(float ior) {
	return sqr(1.0 - 2.0/(ior + 1.0));
}

// Source: https://schuttejoe.github.io/post/disneybsdf/#Sheen
vec3 specularFresnel(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 baseColor) {
    float HdotV = abs(dot(halfwayVector, viewVector));
    float LdotH = dot(lightVector, halfwayVector);

    vec3 tint = calculateTint(baseColor);

    // -- See section 3.1 and 3.2 of the 2015 PBR presentation + the Disney BRDF explorer (which does their
    // -- 2012 remapping rather than the SchlickR0FromRelativeIOR seen here but they mentioned the switch in 3.2).
    vec3 R0 = schlickR0FromRelativeIOR(parameters.ior) * mix(vec3(1.0), tint, parameters.specularTint);
	R0 = mix(R0, baseColor, parameters.metallic);

    float dielectricFresnel = fresnelDielectric(HdotV, 1.0, parameters.ior);

	float fsw0 = (1.0 - LdotH) * (1.0 - LdotH);
	float fresnelSchlickWeight = fsw0 * fsw0 * (1.0-LdotH);
    vec3 metallicFresnel = mix(R0,vec3(1.0),fresnelSchlickWeight);

    return mix(vec3(dielectricFresnel), metallicFresnel, parameters.metallic);
}


vec3 evaluateSpecular(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector, vec3 baseColor, out float samplingPDF, flags hitFlags)
{
	float NdotL = dot(lightVector, normalVector);
    float NdotV = dot(normalVector, viewVector);
    float VdotH = dot(viewVector, halfwayVector);
    float NdotH = dot(normalVector, halfwayVector);
	
    float sinThetaH = sqrt(1.0 - (min(NdotH * NdotH, 0.95)));


	if(NdotL <= 0.0 || NdotV <= 0.0) {
		return vec3(0.0);
	}

	// Anisotropic parameters
	float aspect = sqrt(1.0 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(parameters.roughness) / aspect);
    float ay = max(0.001, sqr(parameters.roughness) * aspect);

	// Microfacet Terms

	float d = anisoGgxD(halfwayVector, normalVector, ax, ay);
	float gl = anisoHeitzGgxG(lightVector, halfwayVector, ax, ay);
	float gv = anisoHeitzGgxG(viewVector, halfwayVector, ax, ay);
	
	vec3 f = specularFresnel(lightVector, halfwayVector, viewVector, baseColor);

	// TODO: Check if / (4.0 * NdotL * NdotV) is needed (I believe not)
    if (hitFlags.specularHit && !hitFlags.clearcoatHit && !hitFlags.transmissionHit) {
        samplingPDF = d * NdotH;
        return f * gl * gv * 4.0 * NdotL * VdotH * sinThetaH / NdotH;
    }
    return d * f * gl * gv * 4.0 * NdotL * VdotH * sinThetaH;
}

// 0.4 Specular Transmission lobe
// https://schuttejoe.github.io/post/disneybsdf/#Sheen
float ThinTransmissionRoughness(float ior)
{
    // -- Disney scales by (.65 * eta - .35) based on figure 15 of the 2015 PBR course notes. Based on their figure
    // -- the results match a geometrically thin solid fairly well.
    return clamp((0.65f * ior - 0.35f) * parameters.roughness, 0.0, 1.0);
}

vec3 evaluateSpecularTransmission(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector, vec3 baseColor, float ax, float ay, out float samplingPDF, flags hitFlags)
{
	// The same ior is needed for refraction and reflection
	// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf

	// TODO: Check if IOR should rather be a seperate parameter
    float relIor = parameters.ior;
    float n2 = relIor * relIor;

    float absNdotL = abs(dot(normalVector, lightVector));
    float absNdotV = abs(dot(normalVector, viewVector));
    float LdotH = dot(halfwayVector, lightVector);
    float VdotH = dot(halfwayVector, viewVector);
    float absLdotH = abs(LdotH);
    float absVdotH = abs(VdotH);

    float d = anisoGgxD(halfwayVector, normalVector, ax, ay);
    float gl = anisoHeitzGgxG(lightVector, halfwayVector, ax, ay);
    float gv = anisoHeitzGgxG(viewVector, halfwayVector, ax, ay);

	// TODP: Implement f
	// Fresnel Term: https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
    float f = fresnelDielectric(VdotH, 1.0, 1.0 / relIor);

    vec3 col;

    if (parameters.thin)
    {
        col = sqrt(baseColor);
    }
    else
    {
        col = baseColor;
    }

    float c = (absLdotH * absVdotH) / (absNdotL * absNdotV);
    float t = (n2 / sqr(LdotH + relIor * VdotH));
    
    float NdotH = dot(normalVector, halfwayVector);
	
    float sinThetaH = sqrt(1.0 - (min(NdotH * NdotH, 0.95)));
    
    if (hitFlags.transmissionHit)
    {
        samplingPDF = dot(lightVector, halfwayVector);
        return (col * c * t * (1.0 - f) * gv * 4.0 * absNdotL * absNdotL * VdotH * sinThetaH) / samplingPDF;
    }
	
    return col * c * t * (1.0 - f) * d * gl * gv;
}

// 0.5 Diffuse BRDF lobe
// Source: https://schuttejoe.github.io/post/disneybsdf/#Sheen
float evaluateDiffuse(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector, vec3 baseColor, out float samplingPDF, flags hitFlags) {
	float NdotL = abs(dot(normalVector, lightVector));
	float NdotV = abs(dot(normalVector, viewVector));
	float HdotL = dot(halfwayVector, lightVector);


	float fl0 = (1.0 - NdotL) * (1.0 - NdotL);
	float fl = fl0*fl0*(1.0 - NdotL);

	float fv0 = (1.0 - NdotV) * (1.0 - NdotV);
	float fv = fv0 * fv0 * (1.0 - NdotV);

	float hanrahanKrueger = 0.0;

	if(parameters.thin && parameters.flatness > 0.0) {
		float r = parameters.roughness * parameters.roughness;

		float fss90 = HdotL * HdotL * r;
		float fss = mix(1.0,fss90,fl) * mix(1.0,fss90,fv);

		float subsurfaceScattering = 1.25 * (fss * (1.0 / (NdotL + NdotV) - 0.5) + 0.5);
		hanrahanKrueger = subsurfaceScattering;
	}

	float lambert = 1.0;
	float rr = 2 * parameters.roughness * HdotL * HdotL;
	float retro = (1.0 / M_PI) * rr * (fl + fv - fl * fv * (rr - 1.0));
	
	// TODO: Replace subsurfaceAprrox with real Subsurface Scattering
	float subsurfaceApproximation = mix(lambert, hanrahanKrueger, parameters.thin ? parameters.flatness : 0.0);
    float cosThetaH = dot(halfwayVector, normalVector);
    float sinThetaH = sqrt(1.0 - min(cosThetaH * cosThetaH, 0.95));
    
    if (!hitFlags.specularHit) {
        samplingPDF = (1.0 / M_PI) * cosThetaH * sinThetaH;
        return (retro + subsurfaceApproximation * (1.0 - 0.5 * fl) * (1.0 - 0.5 * fv)) / (cosThetaH * sinThetaH);
    }

    return (1.0 / M_PI) * (retro + subsurfaceApproximation * (1.0 - 0.5 * fl) * (1.0 - 0.5 * fv));
}

// 1. Sampling

vec3 sampleClearcoat(vec3 viewVector, mat3 frameMatrix) {
    // Adapated from derviations here: https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();
    float alpha = mix(.1, .001, parameters.clearcoatGloss);
    float alpha2 = alpha * alpha;

    // Compute spherical angles
    float theta = acos(sqrt((1 - pow(alpha2, (1- u)))/(1-alpha2)));
    float phi = 2 * M_PI * v;
    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*avoidZero(dot(viewVector,halfwayVector),0.001)*halfwayVector - viewVector;
    
    return lightVector;
}

vec3 sampleSpecular(vec3 viewVector, mat3 frameMatrix) {
    // Adapated from derviations here: https://www.youtube.com/watch?v=xFsJMUS94Fs
    float u = clamp(random(),0.05, 0.95);
    float v = clamp(random(),0.05, 0.95);
    float aspect = sqrt(1.0 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(parameters.roughness) / aspect);
    float ay = max(0.001, sqr(parameters.roughness) * aspect);
    
    float phi = atan2(ay * tan(2*M_PI*u),ax);
    float theta = acos(sqrt((1-v)/(1+((cos(phi)*cos(phi)/(ax*ax))+(sin(phi)*sin(phi)/(ay*ay)))*v)));
    // Problem:
    // sqrt macht Problem für negativ
    // NMormalize macht problem für nahe 0
    vec3 tangentVector = frameMatrix[0];
    vec3 bitangentVector = frameMatrix[1];
    vec3 normalVector = frameMatrix[2];
    vec3 halfwayVector = normalize(sqrt((v)/(1-v))*(ax*cos(2*M_PI*u)*tangentVector + ay*sin(2*M_PI*u)*bitangentVector) + normalVector);

    vec3 lightVector = 2*avoidZero(dot(viewVector,halfwayVector),0.001)*halfwayVector - viewVector;

    return lightVector;
}

vec3 sampleSpecularTransmission(vec3 viewVector, mat3 frameMatrix) {
	/* // We sample this lobe using the distribution of visible normals, proposed by Heitz (2014)
	// https://inria.hal.science/hal-00996995v2/file/slides.pdf
	float rscaled = parameters.thin ? ThinTransmissionRoughness(parameters.ior) : parameters.roughness;
	
	float taspect = sqrt(1.0 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(rscaled) / taspect);
    float ay = max(0.001, sqr(rscaled) * taspect);

	// https://github.com/schuttejoe/Selas/blob/56a7fab5a479ec93d7f641bb64b8170f3b0d3095/Source/Core/Shading/Ggx.cpp#L105-L126
	// -- Stretch the view vector so we are sampling as though roughness==1
	float u1 = random();
	float u2 = random();
    vec3 viewVectorT = transpose(frameMatrix) * viewVector;
    vec3 v = normalize(vec3(viewVectorT.x * ax, viewVectorT.y*ay, viewVectorT.z));

    // -- Build an orthonormal basis with v, t1, and t2
    vec3 t1 = (v.y < 0.9999) ? normalize(cross(v, vec3(0.0,1.0,0.0))) : vec3(1.0,0.0,0.0);
    vec3 t2 = cross(t1, v);
    
    float lensq = v.x * v.x + v.y * v.y;
    vec3 T1 = lensq > 0.001 ? vec3(-v.y, v.x, 0.0) * (1.0 / sqrt(lensq)) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(v, T1);

    // -- Choose a point on a disk with each half of the disk weighted proportionally to its projection onto direction v
    float r = sqrt(u1);
    float phi = 2.0 * M_PI * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + v.z);
    p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;
    
    // -- Calculate the normal in this stretched tangent space
    vec3 n = p1 * T1 + p2 * T2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * v;
    vec3 ne = frameMatrix * (normalize(vec3(ax * n.x, ay * n.y, max(0.0, n.z))));
        
    vec3 lightVector = reflect(viewVector, ne);
    lightVector = -vec3(lightVector.x,lightVector.y,lightVector.z);
    
    // -- unstretch and normalize the normal */
    	// We sample this lobe using the distribution of visible normals, proposed by Heitz (2014)
	
    
    // https://inria.hal.science/hal-00996995v2/file/slides.pdf
    /**float rscaled = parameters.thin ? ThinTransmissionRoughness(parameters.ior) : parameters.roughness;
	
    float taspect = sqrt(1.0 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(rscaled) / taspect);
    float ay = max(0.001, sqr(rscaled) * taspect);

	// https://github.com/schuttejoe/Selas/blob/56a7fab5a479ec93d7f641bb64b8170f3b0d3095/Source/Core/Shading/Ggx.cpp#L105-L126
	// -- Stretch the view vector so we are sampling as though roughness==1
    float u1 = random();
    float u2 = random();
    vec3 viewVectorT = transpose(frameMatrix)*viewVector;
    vec3 v = normalize(vec3(viewVectorT.x * ax, viewVectorT.y*ay, viewVectorT.z));

    // -- Build an orthonormal basis with v, t1, and t2
    vec3 t1 = (v.z < 0.9999) ? normalize(cross(v, vec3(0.0, 0.0, 1.0))) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(t1, v);

    // -- Choose a point on a disk with each half of the disk weighted proportionally to its projection onto direction v
    float a = 1.0 / (1.0 + v.z);
    float r = sqrt(u1);
    float phi = (u2 < a) ? (u2 / a) * M_PI : M_PI + (u2 - a) / (1.0 - a) * M_PI;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi) * ((u2 < a) ? 1.0 : v.z);

    // -- Calculate the normal in this stretched tangent space
    vec3 n = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * v;
    
    vec3 wm = frameMatrix*normalize(vec3(ax * n.x, ay * n.y, n.z));
    
    // Cases: Fresnel Term for the decision: reflect or refract
    // if refract: Transmit
    // if 
    
    //vec3 lightVector = (viewVector, wm, ior);
    //lightVector = vec3(lightVector.x, lightVector.y, -lightVector.z);

    // -- unstretch and normalize the normal
    vec3 lightVector = refract(viewVector, frameMatrix[2], 1.0); **/
    vec3 wi = viewVector; // Transform view vector to tangent space
    float rscaled = parameters.thin ? ThinTransmissionRoughness(parameters.ior) : parameters.roughness;

// Anisotropy and roughness calculations
    float taspect = sqrt(1.0 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(rscaled) / taspect);
    float ay = max(0.001, sqr(rscaled) * taspect);
    vec2 alpha = vec2(ax, ay);

// Normalize the incoming direction (wi) in tangent space
    vec3 wiStd = normalize(wi);

// VNDF Sampling in tangent space
    vec2 u = vec2(random(), random());
    float phi = 2.0f * M_PI * u.x;
    float z = fma((1.0f - u.y), (1.0f + wiStd.y), -wiStd.y);
    float sinTheta = sqrt(clamp(1.0f - z * z, 0.0f, 1.0f));
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    vec3 c = vec3(x, z, y); // Sampled vector in tangent space

// Construct the half-vector in tangent space
    vec3 hStd = c + wiStd;

// Apply anisotropy (stretching) in tangent space
    vec3 wmTangent = normalize(vec3(hStd.x * ax, hStd.y, hStd.z*ay));

// Transform half-vector back to world space
    vec3 wmWorld = wmTangent; // Inverse transform to world space

// Reflect the view vector around wm in world space
    vec3 lightVector = reflect(wiStd, wmWorld);

// Correct the final reflection vector (Z-axis flip for Vulkan)
    return normalize(vec3(-lightVector.x, lightVector.y, -lightVector.z)); // Flip Z-axis only


}

vec3 sampleDiffuse(vec3 viewVector, mat3 frameMatrix) {
    // Source: https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();

    // Compute spherical angles
    float theta = asin(sqrt(u));
    float phi = 2 * M_PI * v;

    vec3 lightVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

	float p = random();
	if(p <= parameters.diffTrans) {
		lightVector = -lightVector;

		// Todo: Check what extinction means and isotropic medium phase function
	}

    return lightVector;
}


vec3 sampleBrdf(vec3 viewVector, vec3 normalVector, mat3 frameMatrix, out flags hitFlags) {
    float metallicBRDF   = parameters.metallic;
    float specularBSDF   = (1.0 - parameters.metallic) * parameters.specTrans;
    float dielectricBRDF = (1.0 - parameters.specTrans) * (1.0 - parameters.metallic);

    float specularWeight     = metallicBRDF + dielectricBRDF;
    float transmissionWeight = specularBSDF;
    float diffuseWeight      = dielectricBRDF;
    float clearcoatWeight    = 1.0 * clamp(parameters.clearcoat, 0.0, 1.0); 

    float norm = 1.0 / (specularWeight + transmissionWeight + diffuseWeight + clearcoatWeight);

    float pSpecular  = specularWeight     * norm;
    float pSpecTrans = transmissionWeight * norm;
    float pDiffuse   = diffuseWeight      * norm;
    float pClearcoat = clearcoatWeight    * norm;

	float u = random();

	if (u < pSpecular) {
		// Sample Specular Lobe
        hitFlags.specularHit = true;
        hitFlags.clearcoatHit = false;
		hitFlags.transmissionHit = false;

		return sampleSpecular(viewVector, frameMatrix);
    } else if (u < pSpecular + pSpecTrans) {
		// Sample Transmission Lobe
        hitFlags.specularHit = true;
        hitFlags.clearcoatHit = false;
		hitFlags.transmissionHit = true;

		return sampleSpecularTransmission(viewVector, frameMatrix);
    } else if (u < pSpecular + pSpecTrans + pClearcoat) {
		// Sample Clearcoat Lobe
        hitFlags.specularHit = true;
        hitFlags.clearcoatHit = true;
		hitFlags.transmissionHit = false;

		return sampleSpecular(viewVector, frameMatrix);
    } else {
		// Sample diffuse lobe
		hitFlags.specularHit = false;
        hitFlags.clearcoatHit = false;
		hitFlags.transmissionHit = false;

		return sampleDiffuse(viewVector, frameMatrix);
	}
}

// 2. Evaluation

vec3 evaluateBrdfPdf() {
    return vec3(0.0);
}

vec3 evaluateBrdf(vec3 lightVector, vec3 viewVector, vec3 normalVector, vec3 baseColor, flags hitFlags) {
	vec3 halfwayVector = normalize(lightVector + viewVector);

	float NdotV = dot(normalVector, viewVector);
	float NdotL = dot(normalVector, lightVector);
    float samplingPDF;

	// Dielectric & Subsurface Scattering & Sheen Lobes
	vec3 sheen = evaluateSheen(lightVector, halfwayVector, viewVector, baseColor);
	vec3 diffuse = evaluateDiffuse(lightVector, halfwayVector, viewVector, normalVector, baseColor, samplingPDF, hitFlags) * baseColor + sheen;
	
	// Tranmission Lobe
	float rscaled = parameters.thin ? ThinTransmissionRoughness(parameters.ior) : parameters.roughness;
	
	float taspect = sqrt(1.0 - parameters.anisotropic * 0.9);
    float tax = max(0.001, sqr(rscaled) / taspect);
    float tay = max(0.001, sqr(rscaled) * taspect);

    vec3 transmission = evaluateSpecularTransmission(lightVector, halfwayVector, viewVector, normalVector, baseColor, tax, tay, samplingPDF, hitFlags);
	
	// Metallic Lobe
    vec3 specularBRDF = evaluateSpecular(lightVector, halfwayVector, viewVector, normalVector, baseColor, samplingPDF, hitFlags);

	// Clearcoat Lobe
    float clearcoat = evaluateClearcoat(lightVector, halfwayVector, viewVector, normalVector, samplingPDF, hitFlags);

    return transmission;
}

// --------- BRDF PUBLIC INTERFACE --------------

// Combined Call to importance sample and evaluate BRDF but not yet cancel out samplingPDF factors

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {
    return vec3(0.0);

}

// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    //vec3 viewVector = -v;
    lightVector = sampleBrdf(viewVector, normalVector, frame, hitFlags);

    return evaluateBrdf(lightVector, viewVector, normalVector, isoSurfaceColor, hitFlags);
}