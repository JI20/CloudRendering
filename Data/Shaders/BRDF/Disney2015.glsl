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

float evaluateClearcoat(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector) {
	if(parameters.clearcoat <= 0.0) {
		return 0.0;
	}

	float NdotH = abs(dot(normalVector, halfwayVector));
	float NdotL = abs(dot(normalVector, lightVector));
	float NdotV = abs(dot(normalVector, viewVector));
	float LdotH = dot(lightVector, halfwayVectorVector);

	float fsw0 = (1.0 - LdotH) * (1.0 - LdotH);
	float fresnelSchlickWeight = fsw0 * fsw0 * (1.0-LdotH);

	float d = GTR1(NdotH, mix(0.1, 0.001, parameters.clearcoatGloss));
	float f = mix(0.04,1.0,fresnelSchlickWeight);
	float gL = smithG_GGX(lightVector, 0.25);
	float gV = smithG_GGX(viewVector, 0.25);

	// TODO: * 0.25 right?
	return 0.25*parameters.clearcoat * d * f * gl * gv;
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
float anisoHeitzGgxG(vec3 vector, vec3 halfwayVector, float ax, float ay)
{
    float dotH = Dot(vector, halfwayVector);
    if (dotH <= 0.0) {
        return 0.0;
    }

    float absTanTheta = Absf(TanTheta(w));
    if(isinf(absTanTheta)) {
        return 0.0;
    }

	// Todo: Find out if vector.x is correct or if it should be lightVector.x
    float lambda = 0.5 * (-1.0 + sqrt(1.0 + (vector.x*vector.x*ax*ax + vector.y*vector.y*ay*ay)/(vector.z*vector.z)));
    return 1.0 / (1.0 + lambda);
}

float schlickR0FromRelativeIOR(float ior) {
	return sqr(1.0 - 2.0/(ior + 1.0));
}

// Source: https://schuttejoe.github.io/post/disneybsdf/#Sheen
vec3 specularFresnel(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 baseColor) {
    float HdotV = abs(dot(halfwayVector, viewVector));

    vec3 tint = calculateTint(baseColor);

    // -- See section 3.1 and 3.2 of the 2015 PBR presentation + the Disney BRDF explorer (which does their
    // -- 2012 remapping rather than the SchlickR0FromRelativeIOR seen here but they mentioned the switch in 3.2).
    vec3 R0 = schlickR0FromRelativeIOR(parameters.ior) * mix(vec3(1.0), tint, parameters.specularTint);
	R0 = mix(R0, baseColor, parameters.metallic);

    float dielectricFresnel = fresnelDielectric(HdotV, 1.0, surface.ior);

	float fsw0 = (1.0 - LdotH) * (1.0 - LdotH);
	float fresnelSchlickWeight = fsw0 * fsw0 * (1.0-LdotH);
    vec3 metallicFresnel = mix(R0,vec3(1.0),fresnelSchlickWeight);

    return mix(vec3(dielectricFresnel), metallicFresnel, parameters.metallic);
}


vec3 evaluateSpecular(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector, vec3 baseColor) {
	float NdotL = dot(lightVector, normalVector);
	float NdotV = dot(normalVector, viewVector)

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
	return d * f * gl * gv;
}

// 0.4 Specular Transmission lobe
// https://schuttejoe.github.io/post/disneybsdf/#Sheen
float ThinTransmissionRoughness(float ior)
{
    // -- Disney scales by (.65 * eta - .35) based on figure 15 of the 2015 PBR course notes. Based on their figure
    // -- the results match a geometrically thin solid fairly well.
    return clamp((0.65f * ior - 0.35f) * parameters.roughness, 0.0, 1.0);
}

// Source: PBR Book v3: https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    // Potentially swap indices of refraction
    // TODO: See if swap is correct
	bool entering = cosThetaI > 0.0;
    if (!entering) {
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
    if (sinThetaT >= 1.0) return 1.0;
    float cosThetaT = sqrt(max(0.0, 1.0 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                  ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                  ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2.0;
}

vec3 evaluateSpecularTransmission(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector, vec3 baseColor, float ax, float ay) {
	// The same ior is needed for refraction and reflection
	// https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf

	// TODO: Check if IOR should rather be a seperate parameter
	float relIor = parameters.ior;
	float n2 = relIor * relIor;

	vec3 absNdotL = abs(dot(normalVector, lightVector));
	vec3 absNdotV = abs(dot(normalVector, viewVector));
	vec3 LdotH = dot(halfwayVector, lightVector);
	vec3 VdotH = dot(halfwayVector, viewVector);
	vec3 absLdotH = abs(LdotH);
	vec3 absVdotH = abs(VdotH);

	float d = anisoGgxD(halfwayVector, ax, ay);
	float gl = anisoHeitzGgxG(lightVector, halfwayVector, ax, ay);
	float gv = anisoHeitzGgxG(viewVector, halfwayVector, ax, ay);

	// TODP: Implement f
	// Fresnel Term: https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
	float f = fresnelDielectric(VdotH, 1.0, 1.0 / relIor);

	vec3 col;

	if(parameters.thin) {
		col = sqrt(baseColor);
	} else {
		col = baseColor;
	}

	float c = (absLdotH * absVdotH) / (absNdotL * absNdotV);
	float t = (n2 / sqr(LdotH + relIor * VdotH));
	
	return col * c * t * (1.0 - f) * d * gl * gv;
}

// 0.5 Diffuse BRDF lobe
// Source: https://schuttejoe.github.io/post/disneybsdf/#Sheen
float evaluateDiffuse(vec3 lightVector, vec3 halfwayVector, vec3 viewVector, vec3 normalVector, vec3 baseColor) {
	float NdotL = abs(dot(normalVector, lightVector));
	float NdotV = abs(dot(normalVector, viewVector));
	float HdotL = dot(halfwayVector, lightVector);


	float fl0 = (1.0 - NdotL) * (1.0 - NdotL);
	float fl = fl0*flo*(1.0 - NdotL);

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

	return (1.0 / M_PI) * (retro + subsurfaceApproximation * (1.0 - 0.5*fl) * (1.0 - 0.5fv));
}

// 1. Sampling

vec3 sampleBrdf(vec3 viewVector, vec3 normalVector, out flags hitFlags) {
    float metallicBRDF   = paramters.metallic;
    float specularBSDF   = (1.0 - parameters.metallic) * parameters.specTrans;
    float dielectricBRDF = (1.0 - parameters.specTrans) * (1.0 - parameters.metallic);

    float specularWeight     = metallicBRDF + dielectricBRDF;
    float transmissionWeight = specularBSDF;
    float diffuseWeight      = dielectricBRDF;
    float clearcoatWeight    = 1.0 * clamp(surface.clearcoat, 0.0, 1.0); 

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
		hitFlags.tansmissionHit = false;

    } else if (u < pSpecular + pSpecTrans) {
		// Sample Transmission Lobe
        hitFlags.specularHit = true;
        hitFlags.clearcoatHit = false;
		hitFlags.tansmissionHit = true;


    } else if (u < pSpecular + pSpecTrans + pClearcoat) {
		// Sample Clearcoat Lobe
        hitFlags.specularHit = true;
        hitFlags.clearcoatHit = true;
		hitFlags.tansmissionHit = false;


    } else {
		// Sample diffuse lobe
		hitFlags.specularHit = false;
        hitFlags.clearcoatHit = false;
		hitFlags.tansmissionHit = false;

	}
}

// 2. Evaluation

vec3 evaluateBrdfPdf() {

}

vec3 evaluateBrdf(vec3 lightVector, vec3 viewVector, vec3 normalVector, vec3 baseColor) {
	vec3 halfwayVector = normalize(lightVector + viewVector);

	float NdotV = dot(normalVector, viewVector);
	float NdotL = dot(normalVector, lightVector);

	// Dielectric & Subsurface Scattering & Sheen Lobes
	vec3 sheen = evaluateSheen(lightVector, halfwayVector, viewVector, baseColor);
	vec3 diffuse = evaluateDiffuse(lightVector, halfwayVector, viewVector, normalVector, baseColor) * baseColor + sheen;
	
	// Tranmission Lobe
	float rscaled = parameters.thin ? ThinTransmissionRoughness(parameters.ior) : parameters.roughness;
	
	float taspect = sqrt(1.0 - parameters.anisotropic * 0.9);
    float tax = max(0.001, sqr(rscaled) / taspect);
    float tay = max(0.001, sqr(rscaled) * taspect);

	vec3 transmission = evaluateSpecularTransmission(lightVector, halfwayVector, viewVector, normalVector, baseColor, tax, tay);
	
	// Metallic Lobe
	vec3 specularBRDF = evaluateSpecular(lightVector, halfwayVector, viewVector, normalVector, baseColor);

	// Clearcoat Lobe
	vec3 clearcoat = evaluateClearcoat(lightVector, halfwayVector, viewVector, normalVector);

	return diffuse + transmission + specularBRDF + clearcoat;
}

// --------- BRDF PUBLIC INTERFACE --------------

// Combined Call to importance sample and evaluate BRDF but not yet cancel out samplingPDF factors

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {

}

// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {

}