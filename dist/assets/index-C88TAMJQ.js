(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))n(r);new MutationObserver(r=>{for(const s of r)if(s.type==="childList")for(const a of s.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&n(a)}).observe(document,{childList:!0,subtree:!0});function t(r){const s={};return r.integrity&&(s.integrity=r.integrity),r.referrerPolicy&&(s.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?s.credentials="include":r.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function n(r){if(r.ep)return;r.ep=!0;const s=t(r);fetch(r.href,s)}})();const io="183",zi={ROTATE:0,DOLLY:1,PAN:2},Bi={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},Rc=0,bo=1,Pc=2,rs=1,Dc=2,ur=3,ri=0,qt=1,Bn=2,zn=0,Gi=1,To=2,Ao=3,wo=4,Ic=5,fi=100,Lc=101,Uc=102,Fc=103,Nc=104,Oc=200,Bc=201,kc=202,zc=203,oa=204,la=205,Gc=206,Hc=207,Vc=208,Wc=209,Xc=210,Yc=211,qc=212,Zc=213,$c=214,ca=0,ha=1,ua=2,Vi=3,da=4,fa=5,pa=6,ma=7,ro=0,jc=1,Kc=2,An=0,Gl=1,Hl=2,Vl=3,Wl=4,Xl=5,Yl=6,ql=7,Zl=300,_i=301,Wi=302,ss=303,ys=304,gs=306,ga=1e3,gn=1001,_a=1002,Nt=1003,Jc=1004,wr=1005,bt=1006,Es=1007,ti=1008,Kt=1009,$l=1010,jl=1011,mr=1012,so=1013,Cn=1014,Yt=1015,Jt=1016,ao=1017,oo=1018,gr=1020,Kl=35902,Jl=35899,Ql=1021,ec=1022,Ft=1023,Hn=1026,mi=1027,gi=1028,lo=1029,mn=1030,co=1031,ho=1033,as=33776,os=33777,ls=33778,cs=33779,xa=35840,va=35841,Sa=35842,Ma=35843,ya=36196,Ea=37492,ba=37496,Ta=37488,Aa=37489,wa=37490,Ca=37491,Ra=37808,Pa=37809,Da=37810,Ia=37811,La=37812,Ua=37813,Fa=37814,Na=37815,Oa=37816,Ba=37817,ka=37818,za=37819,Ga=37820,Ha=37821,Va=36492,Wa=36494,Xa=36495,Ya=36283,qa=36284,Za=36285,$a=36286,Qc=3200,tc=0,eh=1,ei="",nn="srgb",jt="srgb-linear",us="linear",ft="srgb",yi=7680,Co=519,th=512,nh=513,ih=514,uo=515,rh=516,sh=517,fo=518,ah=519,Ro=35044,oh=35048,Po="300 es",Tn=2e3,_r=2001;function lh(i){for(let e=i.length-1;e>=0;--e)if(i[e]>=65535)return!0;return!1}function ds(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function ch(){const i=ds("canvas");return i.style.display="block",i}const Do={};function Io(...i){const e="THREE."+i.shift();console.log(e,...i)}function nc(i){const e=i[0];if(typeof e=="string"&&e.startsWith("TSL:")){const t=i[1];t&&t.isStackTrace?i[0]+=" "+t.getLocation():i[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return i}function Xe(...i){i=nc(i);const e="THREE."+i.shift();{const t=i[0];t&&t.isStackTrace?console.warn(t.getError(e)):console.warn(e,...i)}}function ot(...i){i=nc(i);const e="THREE."+i.shift();{const t=i[0];t&&t.isStackTrace?console.error(t.getError(e)):console.error(e,...i)}}function fs(...i){const e=i.join(" ");e in Do||(Do[e]=!0,Xe(...i))}function hh(i,e,t){return new Promise(function(n,r){function s(){switch(i.clientWaitSync(e,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:r();break;case i.TIMEOUT_EXPIRED:setTimeout(s,t);break;default:n()}}setTimeout(s,t)})}const uh={[ca]:ha,[ua]:pa,[da]:ma,[Vi]:fa,[ha]:ca,[pa]:ua,[ma]:da,[fa]:Vi};class xi{addEventListener(e,t){this._listeners===void 0&&(this._listeners={});const n=this._listeners;n[e]===void 0&&(n[e]=[]),n[e].indexOf(t)===-1&&n[e].push(t)}hasEventListener(e,t){const n=this._listeners;return n===void 0?!1:n[e]!==void 0&&n[e].indexOf(t)!==-1}removeEventListener(e,t){const n=this._listeners;if(n===void 0)return;const r=n[e];if(r!==void 0){const s=r.indexOf(t);s!==-1&&r.splice(s,1)}}dispatchEvent(e){const t=this._listeners;if(t===void 0)return;const n=t[e.type];if(n!==void 0){e.target=this;const r=n.slice(0);for(let s=0,a=r.length;s<a;s++)r[s].call(this,e);e.target=null}}}const Bt=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],fr=Math.PI/180,ja=180/Math.PI;function vr(){const i=Math.random()*4294967295|0,e=Math.random()*4294967295|0,t=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(Bt[i&255]+Bt[i>>8&255]+Bt[i>>16&255]+Bt[i>>24&255]+"-"+Bt[e&255]+Bt[e>>8&255]+"-"+Bt[e>>16&15|64]+Bt[e>>24&255]+"-"+Bt[t&63|128]+Bt[t>>8&255]+"-"+Bt[t>>16&255]+Bt[t>>24&255]+Bt[n&255]+Bt[n>>8&255]+Bt[n>>16&255]+Bt[n>>24&255]).toLowerCase()}function nt(i,e,t){return Math.max(e,Math.min(t,i))}function dh(i,e){return(i%e+e)%e}function bs(i,e,t){return(1-t)*i+t*e}function Ji(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return i/4294967295;case Uint16Array:return i/65535;case Uint8Array:return i/255;case Int32Array:return Math.max(i/2147483647,-1);case Int16Array:return Math.max(i/32767,-1);case Int8Array:return Math.max(i/127,-1);default:throw new Error("Invalid component type.")}}function Vt(i,e){switch(e.constructor){case Float32Array:return i;case Uint32Array:return Math.round(i*4294967295);case Uint16Array:return Math.round(i*65535);case Uint8Array:return Math.round(i*255);case Int32Array:return Math.round(i*2147483647);case Int16Array:return Math.round(i*32767);case Int8Array:return Math.round(i*127);default:throw new Error("Invalid component type.")}}const fh={DEG2RAD:fr};class $e{constructor(e=0,t=0){$e.prototype.isVector2=!0,this.x=e,this.y=t}get width(){return this.x}set width(e){this.x=e}get height(){return this.y}set height(e){this.y=e}set(e,t){return this.x=e,this.y=t,this}setScalar(e){return this.x=e,this.y=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y)}copy(e){return this.x=e.x,this.y=e.y,this}add(e){return this.x+=e.x,this.y+=e.y,this}addScalar(e){return this.x+=e,this.y+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this}subScalar(e){return this.x-=e,this.y-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this}multiply(e){return this.x*=e.x,this.y*=e.y,this}multiplyScalar(e){return this.x*=e,this.y*=e,this}divide(e){return this.x/=e.x,this.y/=e.y,this}divideScalar(e){return this.multiplyScalar(1/e)}applyMatrix3(e){const t=this.x,n=this.y,r=e.elements;return this.x=r[0]*t+r[3]*n+r[6],this.y=r[1]*t+r[4]*n+r[7],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this}clamp(e,t){return this.x=nt(this.x,e.x,t.x),this.y=nt(this.y,e.y,t.y),this}clampScalar(e,t){return this.x=nt(this.x,e,t),this.y=nt(this.y,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(nt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(e){return this.x*e.x+this.y*e.y}cross(e){return this.x*e.y-this.y*e.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(nt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y;return t*t+n*n}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this}equals(e){return e.x===this.x&&e.y===this.y}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this}rotateAround(e,t){const n=Math.cos(t),r=Math.sin(t),s=this.x-e.x,a=this.y-e.y;return this.x=s*n-a*r+e.x,this.y=s*r+a*n+e.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}}class si{constructor(e=0,t=0,n=0,r=1){this.isQuaternion=!0,this._x=e,this._y=t,this._z=n,this._w=r}static slerpFlat(e,t,n,r,s,a,o){let c=n[r+0],l=n[r+1],u=n[r+2],d=n[r+3],h=s[a+0],f=s[a+1],_=s[a+2],y=s[a+3];if(d!==y||c!==h||l!==f||u!==_){let g=c*h+l*f+u*_+d*y;g<0&&(h=-h,f=-f,_=-_,y=-y,g=-g);let m=1-o;if(g<.9995){const b=Math.acos(g),w=Math.sin(b);m=Math.sin(m*b)/w,o=Math.sin(o*b)/w,c=c*m+h*o,l=l*m+f*o,u=u*m+_*o,d=d*m+y*o}else{c=c*m+h*o,l=l*m+f*o,u=u*m+_*o,d=d*m+y*o;const b=1/Math.sqrt(c*c+l*l+u*u+d*d);c*=b,l*=b,u*=b,d*=b}}e[t]=c,e[t+1]=l,e[t+2]=u,e[t+3]=d}static multiplyQuaternionsFlat(e,t,n,r,s,a){const o=n[r],c=n[r+1],l=n[r+2],u=n[r+3],d=s[a],h=s[a+1],f=s[a+2],_=s[a+3];return e[t]=o*_+u*d+c*f-l*h,e[t+1]=c*_+u*h+l*d-o*f,e[t+2]=l*_+u*f+o*h-c*d,e[t+3]=u*_-o*d-c*h-l*f,e}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get w(){return this._w}set w(e){this._w=e,this._onChangeCallback()}set(e,t,n,r){return this._x=e,this._y=t,this._z=n,this._w=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(e){return this._x=e.x,this._y=e.y,this._z=e.z,this._w=e.w,this._onChangeCallback(),this}setFromEuler(e,t=!0){const n=e._x,r=e._y,s=e._z,a=e._order,o=Math.cos,c=Math.sin,l=o(n/2),u=o(r/2),d=o(s/2),h=c(n/2),f=c(r/2),_=c(s/2);switch(a){case"XYZ":this._x=h*u*d+l*f*_,this._y=l*f*d-h*u*_,this._z=l*u*_+h*f*d,this._w=l*u*d-h*f*_;break;case"YXZ":this._x=h*u*d+l*f*_,this._y=l*f*d-h*u*_,this._z=l*u*_-h*f*d,this._w=l*u*d+h*f*_;break;case"ZXY":this._x=h*u*d-l*f*_,this._y=l*f*d+h*u*_,this._z=l*u*_+h*f*d,this._w=l*u*d-h*f*_;break;case"ZYX":this._x=h*u*d-l*f*_,this._y=l*f*d+h*u*_,this._z=l*u*_-h*f*d,this._w=l*u*d+h*f*_;break;case"YZX":this._x=h*u*d+l*f*_,this._y=l*f*d+h*u*_,this._z=l*u*_-h*f*d,this._w=l*u*d-h*f*_;break;case"XZY":this._x=h*u*d-l*f*_,this._y=l*f*d-h*u*_,this._z=l*u*_+h*f*d,this._w=l*u*d+h*f*_;break;default:Xe("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return t===!0&&this._onChangeCallback(),this}setFromAxisAngle(e,t){const n=t/2,r=Math.sin(n);return this._x=e.x*r,this._y=e.y*r,this._z=e.z*r,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(e){const t=e.elements,n=t[0],r=t[4],s=t[8],a=t[1],o=t[5],c=t[9],l=t[2],u=t[6],d=t[10],h=n+o+d;if(h>0){const f=.5/Math.sqrt(h+1);this._w=.25/f,this._x=(u-c)*f,this._y=(s-l)*f,this._z=(a-r)*f}else if(n>o&&n>d){const f=2*Math.sqrt(1+n-o-d);this._w=(u-c)/f,this._x=.25*f,this._y=(r+a)/f,this._z=(s+l)/f}else if(o>d){const f=2*Math.sqrt(1+o-n-d);this._w=(s-l)/f,this._x=(r+a)/f,this._y=.25*f,this._z=(c+u)/f}else{const f=2*Math.sqrt(1+d-n-o);this._w=(a-r)/f,this._x=(s+l)/f,this._y=(c+u)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(e,t){let n=e.dot(t)+1;return n<1e-8?(n=0,Math.abs(e.x)>Math.abs(e.z)?(this._x=-e.y,this._y=e.x,this._z=0,this._w=n):(this._x=0,this._y=-e.z,this._z=e.y,this._w=n)):(this._x=e.y*t.z-e.z*t.y,this._y=e.z*t.x-e.x*t.z,this._z=e.x*t.y-e.y*t.x,this._w=n),this.normalize()}angleTo(e){return 2*Math.acos(Math.abs(nt(this.dot(e),-1,1)))}rotateTowards(e,t){const n=this.angleTo(e);if(n===0)return this;const r=Math.min(1,t/n);return this.slerp(e,r),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(e){return this._x*e._x+this._y*e._y+this._z*e._z+this._w*e._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let e=this.length();return e===0?(this._x=0,this._y=0,this._z=0,this._w=1):(e=1/e,this._x=this._x*e,this._y=this._y*e,this._z=this._z*e,this._w=this._w*e),this._onChangeCallback(),this}multiply(e){return this.multiplyQuaternions(this,e)}premultiply(e){return this.multiplyQuaternions(e,this)}multiplyQuaternions(e,t){const n=e._x,r=e._y,s=e._z,a=e._w,o=t._x,c=t._y,l=t._z,u=t._w;return this._x=n*u+a*o+r*l-s*c,this._y=r*u+a*c+s*o-n*l,this._z=s*u+a*l+n*c-r*o,this._w=a*u-n*o-r*c-s*l,this._onChangeCallback(),this}slerp(e,t){let n=e._x,r=e._y,s=e._z,a=e._w,o=this.dot(e);o<0&&(n=-n,r=-r,s=-s,a=-a,o=-o);let c=1-t;if(o<.9995){const l=Math.acos(o),u=Math.sin(l);c=Math.sin(c*l)/u,t=Math.sin(t*l)/u,this._x=this._x*c+n*t,this._y=this._y*c+r*t,this._z=this._z*c+s*t,this._w=this._w*c+a*t,this._onChangeCallback()}else this._x=this._x*c+n*t,this._y=this._y*c+r*t,this._z=this._z*c+s*t,this._w=this._w*c+a*t,this.normalize();return this}slerpQuaternions(e,t,n){return this.copy(e).slerp(t,n)}random(){const e=2*Math.PI*Math.random(),t=2*Math.PI*Math.random(),n=Math.random(),r=Math.sqrt(1-n),s=Math.sqrt(n);return this.set(r*Math.sin(e),r*Math.cos(e),s*Math.sin(t),s*Math.cos(t))}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._w===this._w}fromArray(e,t=0){return this._x=e[t],this._y=e[t+1],this._z=e[t+2],this._w=e[t+3],this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._w,e}fromBufferAttribute(e,t){return this._x=e.getX(t),this._y=e.getY(t),this._z=e.getZ(t),this._w=e.getW(t),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}}class q{constructor(e=0,t=0,n=0){q.prototype.isVector3=!0,this.x=e,this.y=t,this.z=n}set(e,t,n){return n===void 0&&(n=this.z),this.x=e,this.y=t,this.z=n,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this}multiplyVectors(e,t){return this.x=e.x*t.x,this.y=e.y*t.y,this.z=e.z*t.z,this}applyEuler(e){return this.applyQuaternion(Lo.setFromEuler(e))}applyAxisAngle(e,t){return this.applyQuaternion(Lo.setFromAxisAngle(e,t))}applyMatrix3(e){const t=this.x,n=this.y,r=this.z,s=e.elements;return this.x=s[0]*t+s[3]*n+s[6]*r,this.y=s[1]*t+s[4]*n+s[7]*r,this.z=s[2]*t+s[5]*n+s[8]*r,this}applyNormalMatrix(e){return this.applyMatrix3(e).normalize()}applyMatrix4(e){const t=this.x,n=this.y,r=this.z,s=e.elements,a=1/(s[3]*t+s[7]*n+s[11]*r+s[15]);return this.x=(s[0]*t+s[4]*n+s[8]*r+s[12])*a,this.y=(s[1]*t+s[5]*n+s[9]*r+s[13])*a,this.z=(s[2]*t+s[6]*n+s[10]*r+s[14])*a,this}applyQuaternion(e){const t=this.x,n=this.y,r=this.z,s=e.x,a=e.y,o=e.z,c=e.w,l=2*(a*r-o*n),u=2*(o*t-s*r),d=2*(s*n-a*t);return this.x=t+c*l+a*d-o*u,this.y=n+c*u+o*l-s*d,this.z=r+c*d+s*u-a*l,this}project(e){return this.applyMatrix4(e.matrixWorldInverse).applyMatrix4(e.projectionMatrix)}unproject(e){return this.applyMatrix4(e.projectionMatrixInverse).applyMatrix4(e.matrixWorld)}transformDirection(e){const t=this.x,n=this.y,r=this.z,s=e.elements;return this.x=s[0]*t+s[4]*n+s[8]*r,this.y=s[1]*t+s[5]*n+s[9]*r,this.z=s[2]*t+s[6]*n+s[10]*r,this.normalize()}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this}divideScalar(e){return this.multiplyScalar(1/e)}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this}clamp(e,t){return this.x=nt(this.x,e.x,t.x),this.y=nt(this.y,e.y,t.y),this.z=nt(this.z,e.z,t.z),this}clampScalar(e,t){return this.x=nt(this.x,e,t),this.y=nt(this.y,e,t),this.z=nt(this.z,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(nt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this}cross(e){return this.crossVectors(this,e)}crossVectors(e,t){const n=e.x,r=e.y,s=e.z,a=t.x,o=t.y,c=t.z;return this.x=r*c-s*o,this.y=s*a-n*c,this.z=n*o-r*a,this}projectOnVector(e){const t=e.lengthSq();if(t===0)return this.set(0,0,0);const n=e.dot(this)/t;return this.copy(e).multiplyScalar(n)}projectOnPlane(e){return Ts.copy(this).projectOnVector(e),this.sub(Ts)}reflect(e){return this.sub(Ts.copy(e).multiplyScalar(2*this.dot(e)))}angleTo(e){const t=Math.sqrt(this.lengthSq()*e.lengthSq());if(t===0)return Math.PI/2;const n=this.dot(e)/t;return Math.acos(nt(n,-1,1))}distanceTo(e){return Math.sqrt(this.distanceToSquared(e))}distanceToSquared(e){const t=this.x-e.x,n=this.y-e.y,r=this.z-e.z;return t*t+n*n+r*r}manhattanDistanceTo(e){return Math.abs(this.x-e.x)+Math.abs(this.y-e.y)+Math.abs(this.z-e.z)}setFromSpherical(e){return this.setFromSphericalCoords(e.radius,e.phi,e.theta)}setFromSphericalCoords(e,t,n){const r=Math.sin(t)*e;return this.x=r*Math.sin(n),this.y=Math.cos(t)*e,this.z=r*Math.cos(n),this}setFromCylindrical(e){return this.setFromCylindricalCoords(e.radius,e.theta,e.y)}setFromCylindricalCoords(e,t,n){return this.x=e*Math.sin(t),this.y=n,this.z=e*Math.cos(t),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this}setFromMatrixScale(e){const t=this.setFromMatrixColumn(e,0).length(),n=this.setFromMatrixColumn(e,1).length(),r=this.setFromMatrixColumn(e,2).length();return this.x=t,this.y=n,this.z=r,this}setFromMatrixColumn(e,t){return this.fromArray(e.elements,t*4)}setFromMatrix3Column(e,t){return this.fromArray(e.elements,t*3)}setFromEuler(e){return this.x=e._x,this.y=e._y,this.z=e._z,this}setFromColor(e){return this.x=e.r,this.y=e.g,this.z=e.b,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){const e=Math.random()*Math.PI*2,t=Math.random()*2-1,n=Math.sqrt(1-t*t);return this.x=n*Math.cos(e),this.y=t,this.z=n*Math.sin(e),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}}const Ts=new q,Lo=new si;class Je{constructor(e,t,n,r,s,a,o,c,l){Je.prototype.isMatrix3=!0,this.elements=[1,0,0,0,1,0,0,0,1],e!==void 0&&this.set(e,t,n,r,s,a,o,c,l)}set(e,t,n,r,s,a,o,c,l){const u=this.elements;return u[0]=e,u[1]=r,u[2]=o,u[3]=t,u[4]=s,u[5]=c,u[6]=n,u[7]=a,u[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],this}extractBasis(e,t,n){return e.setFromMatrix3Column(this,0),t.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(e){const t=e.elements;return this.set(t[0],t[4],t[8],t[1],t[5],t[9],t[2],t[6],t[10]),this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,r=t.elements,s=this.elements,a=n[0],o=n[3],c=n[6],l=n[1],u=n[4],d=n[7],h=n[2],f=n[5],_=n[8],y=r[0],g=r[3],m=r[6],b=r[1],w=r[4],A=r[7],U=r[2],L=r[5],N=r[8];return s[0]=a*y+o*b+c*U,s[3]=a*g+o*w+c*L,s[6]=a*m+o*A+c*N,s[1]=l*y+u*b+d*U,s[4]=l*g+u*w+d*L,s[7]=l*m+u*A+d*N,s[2]=h*y+f*b+_*U,s[5]=h*g+f*w+_*L,s[8]=h*m+f*A+_*N,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[3]*=e,t[6]*=e,t[1]*=e,t[4]*=e,t[7]*=e,t[2]*=e,t[5]*=e,t[8]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[1],r=e[2],s=e[3],a=e[4],o=e[5],c=e[6],l=e[7],u=e[8];return t*a*u-t*o*l-n*s*u+n*o*c+r*s*l-r*a*c}invert(){const e=this.elements,t=e[0],n=e[1],r=e[2],s=e[3],a=e[4],o=e[5],c=e[6],l=e[7],u=e[8],d=u*a-o*l,h=o*c-u*s,f=l*s-a*c,_=t*d+n*h+r*f;if(_===0)return this.set(0,0,0,0,0,0,0,0,0);const y=1/_;return e[0]=d*y,e[1]=(r*l-u*n)*y,e[2]=(o*n-r*a)*y,e[3]=h*y,e[4]=(u*t-r*c)*y,e[5]=(r*s-o*t)*y,e[6]=f*y,e[7]=(n*c-l*t)*y,e[8]=(a*t-n*s)*y,this}transpose(){let e;const t=this.elements;return e=t[1],t[1]=t[3],t[3]=e,e=t[2],t[2]=t[6],t[6]=e,e=t[5],t[5]=t[7],t[7]=e,this}getNormalMatrix(e){return this.setFromMatrix4(e).invert().transpose()}transposeIntoArray(e){const t=this.elements;return e[0]=t[0],e[1]=t[3],e[2]=t[6],e[3]=t[1],e[4]=t[4],e[5]=t[7],e[6]=t[2],e[7]=t[5],e[8]=t[8],this}setUvTransform(e,t,n,r,s,a,o){const c=Math.cos(s),l=Math.sin(s);return this.set(n*c,n*l,-n*(c*a+l*o)+a+e,-r*l,r*c,-r*(-l*a+c*o)+o+t,0,0,1),this}scale(e,t){return this.premultiply(As.makeScale(e,t)),this}rotate(e){return this.premultiply(As.makeRotation(-e)),this}translate(e,t){return this.premultiply(As.makeTranslation(e,t)),this}makeTranslation(e,t){return e.isVector2?this.set(1,0,e.x,0,1,e.y,0,0,1):this.set(1,0,e,0,1,t,0,0,1),this}makeRotation(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,n,t,0,0,0,1),this}makeScale(e,t){return this.set(e,0,0,0,t,0,0,0,1),this}equals(e){const t=this.elements,n=e.elements;for(let r=0;r<9;r++)if(t[r]!==n[r])return!1;return!0}fromArray(e,t=0){for(let n=0;n<9;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e}clone(){return new this.constructor().fromArray(this.elements)}}const As=new Je,Uo=new Je().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),Fo=new Je().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function ph(){const i={enabled:!0,workingColorSpace:jt,spaces:{},convert:function(r,s,a){return this.enabled===!1||s===a||!s||!a||(this.spaces[s].transfer===ft&&(r.r=Gn(r.r),r.g=Gn(r.g),r.b=Gn(r.b)),this.spaces[s].primaries!==this.spaces[a].primaries&&(r.applyMatrix3(this.spaces[s].toXYZ),r.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===ft&&(r.r=Hi(r.r),r.g=Hi(r.g),r.b=Hi(r.b))),r},workingToColorSpace:function(r,s){return this.convert(r,this.workingColorSpace,s)},colorSpaceToWorking:function(r,s){return this.convert(r,s,this.workingColorSpace)},getPrimaries:function(r){return this.spaces[r].primaries},getTransfer:function(r){return r===ei?us:this.spaces[r].transfer},getToneMappingMode:function(r){return this.spaces[r].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(r,s=this.workingColorSpace){return r.fromArray(this.spaces[s].luminanceCoefficients)},define:function(r){Object.assign(this.spaces,r)},_getMatrix:function(r,s,a){return r.copy(this.spaces[s].toXYZ).multiply(this.spaces[a].fromXYZ)},_getDrawingBufferColorSpace:function(r){return this.spaces[r].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(r=this.workingColorSpace){return this.spaces[r].workingColorSpaceConfig.unpackColorSpace},fromWorkingColorSpace:function(r,s){return fs("ColorManagement: .fromWorkingColorSpace() has been renamed to .workingToColorSpace()."),i.workingToColorSpace(r,s)},toWorkingColorSpace:function(r,s){return fs("ColorManagement: .toWorkingColorSpace() has been renamed to .colorSpaceToWorking()."),i.colorSpaceToWorking(r,s)}},e=[.64,.33,.3,.6,.15,.06],t=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[jt]:{primaries:e,whitePoint:n,transfer:us,toXYZ:Uo,fromXYZ:Fo,luminanceCoefficients:t,workingColorSpaceConfig:{unpackColorSpace:nn},outputColorSpaceConfig:{drawingBufferColorSpace:nn}},[nn]:{primaries:e,whitePoint:n,transfer:ft,toXYZ:Uo,fromXYZ:Fo,luminanceCoefficients:t,outputColorSpaceConfig:{drawingBufferColorSpace:nn}}}),i}const lt=ph();function Gn(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function Hi(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}let Ei;class mh{static getDataURL(e,t="image/png"){if(/^data:/i.test(e.src)||typeof HTMLCanvasElement>"u")return e.src;let n;if(e instanceof HTMLCanvasElement)n=e;else{Ei===void 0&&(Ei=ds("canvas")),Ei.width=e.width,Ei.height=e.height;const r=Ei.getContext("2d");e instanceof ImageData?r.putImageData(e,0,0):r.drawImage(e,0,0,e.width,e.height),n=Ei}return n.toDataURL(t)}static sRGBToLinear(e){if(typeof HTMLImageElement<"u"&&e instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&e instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&e instanceof ImageBitmap){const t=ds("canvas");t.width=e.width,t.height=e.height;const n=t.getContext("2d");n.drawImage(e,0,0,e.width,e.height);const r=n.getImageData(0,0,e.width,e.height),s=r.data;for(let a=0;a<s.length;a++)s[a]=Gn(s[a]/255)*255;return n.putImageData(r,0,0),t}else if(e.data){const t=e.data.slice(0);for(let n=0;n<t.length;n++)t instanceof Uint8Array||t instanceof Uint8ClampedArray?t[n]=Math.floor(Gn(t[n]/255)*255):t[n]=Gn(t[n]);return{data:t,width:e.width,height:e.height}}else return Xe("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),e}}let gh=0;class po{constructor(e=null){this.isSource=!0,Object.defineProperty(this,"id",{value:gh++}),this.uuid=vr(),this.data=e,this.dataReady=!0,this.version=0}getSize(e){const t=this.data;return typeof HTMLVideoElement<"u"&&t instanceof HTMLVideoElement?e.set(t.videoWidth,t.videoHeight,0):typeof VideoFrame<"u"&&t instanceof VideoFrame?e.set(t.displayHeight,t.displayWidth,0):t!==null?e.set(t.width,t.height,t.depth||0):e.set(0,0,0),e}set needsUpdate(e){e===!0&&this.version++}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.images[this.uuid]!==void 0)return e.images[this.uuid];const n={uuid:this.uuid,url:""},r=this.data;if(r!==null){let s;if(Array.isArray(r)){s=[];for(let a=0,o=r.length;a<o;a++)r[a].isDataTexture?s.push(ws(r[a].image)):s.push(ws(r[a]))}else s=ws(r);n.url=s}return t||(e.images[this.uuid]=n),n}}function ws(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?mh.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(Xe("Texture: Unable to serialize Texture."),{})}let _h=0;const Cs=new q;class Ht extends xi{constructor(e=Ht.DEFAULT_IMAGE,t=Ht.DEFAULT_MAPPING,n=gn,r=gn,s=bt,a=ti,o=Ft,c=Kt,l=Ht.DEFAULT_ANISOTROPY,u=ei){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:_h++}),this.uuid=vr(),this.name="",this.source=new po(e),this.mipmaps=[],this.mapping=t,this.channel=0,this.wrapS=n,this.wrapT=r,this.magFilter=s,this.minFilter=a,this.anisotropy=l,this.format=o,this.internalFormat=null,this.type=c,this.offset=new $e(0,0),this.repeat=new $e(1,1),this.center=new $e(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Je,this.generateMipmaps=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(e&&e.depth&&e.depth>1),this.pmremVersion=0}get width(){return this.source.getSize(Cs).x}get height(){return this.source.getSize(Cs).y}get depth(){return this.source.getSize(Cs).z}get image(){return this.source.data}set image(e=null){this.source.data=e}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(e){return this.name=e.name,this.source=e.source,this.mipmaps=e.mipmaps.slice(0),this.mapping=e.mapping,this.channel=e.channel,this.wrapS=e.wrapS,this.wrapT=e.wrapT,this.magFilter=e.magFilter,this.minFilter=e.minFilter,this.anisotropy=e.anisotropy,this.format=e.format,this.internalFormat=e.internalFormat,this.type=e.type,this.offset.copy(e.offset),this.repeat.copy(e.repeat),this.center.copy(e.center),this.rotation=e.rotation,this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrix.copy(e.matrix),this.generateMipmaps=e.generateMipmaps,this.premultiplyAlpha=e.premultiplyAlpha,this.flipY=e.flipY,this.unpackAlignment=e.unpackAlignment,this.colorSpace=e.colorSpace,this.renderTarget=e.renderTarget,this.isRenderTargetTexture=e.isRenderTargetTexture,this.isArrayTexture=e.isArrayTexture,this.userData=JSON.parse(JSON.stringify(e.userData)),this.needsUpdate=!0,this}setValues(e){for(const t in e){const n=e[t];if(n===void 0){Xe(`Texture.setValues(): parameter '${t}' has value of undefined.`);continue}const r=this[t];if(r===void 0){Xe(`Texture.setValues(): property '${t}' does not exist.`);continue}r&&n&&r.isVector2&&n.isVector2||r&&n&&r.isVector3&&n.isVector3||r&&n&&r.isMatrix3&&n.isMatrix3?r.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";if(!t&&e.textures[this.uuid]!==void 0)return e.textures[this.uuid];const n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(e).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),t||(e.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(e){if(this.mapping!==Zl)return e;if(e.applyMatrix3(this.matrix),e.x<0||e.x>1)switch(this.wrapS){case ga:e.x=e.x-Math.floor(e.x);break;case gn:e.x=e.x<0?0:1;break;case _a:Math.abs(Math.floor(e.x)%2)===1?e.x=Math.ceil(e.x)-e.x:e.x=e.x-Math.floor(e.x);break}if(e.y<0||e.y>1)switch(this.wrapT){case ga:e.y=e.y-Math.floor(e.y);break;case gn:e.y=e.y<0?0:1;break;case _a:Math.abs(Math.floor(e.y)%2)===1?e.y=Math.ceil(e.y)-e.y:e.y=e.y-Math.floor(e.y);break}return this.flipY&&(e.y=1-e.y),e}set needsUpdate(e){e===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(e){e===!0&&this.pmremVersion++}}Ht.DEFAULT_IMAGE=null;Ht.DEFAULT_MAPPING=Zl;Ht.DEFAULT_ANISOTROPY=1;class Et{constructor(e=0,t=0,n=0,r=1){Et.prototype.isVector4=!0,this.x=e,this.y=t,this.z=n,this.w=r}get width(){return this.z}set width(e){this.z=e}get height(){return this.w}set height(e){this.w=e}set(e,t,n,r){return this.x=e,this.y=t,this.z=n,this.w=r,this}setScalar(e){return this.x=e,this.y=e,this.z=e,this.w=e,this}setX(e){return this.x=e,this}setY(e){return this.y=e,this}setZ(e){return this.z=e,this}setW(e){return this.w=e,this}setComponent(e,t){switch(e){case 0:this.x=t;break;case 1:this.y=t;break;case 2:this.z=t;break;case 3:this.w=t;break;default:throw new Error("index is out of range: "+e)}return this}getComponent(e){switch(e){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("index is out of range: "+e)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(e){return this.x=e.x,this.y=e.y,this.z=e.z,this.w=e.w!==void 0?e.w:1,this}add(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this.w+=e.w,this}addScalar(e){return this.x+=e,this.y+=e,this.z+=e,this.w+=e,this}addVectors(e,t){return this.x=e.x+t.x,this.y=e.y+t.y,this.z=e.z+t.z,this.w=e.w+t.w,this}addScaledVector(e,t){return this.x+=e.x*t,this.y+=e.y*t,this.z+=e.z*t,this.w+=e.w*t,this}sub(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this.w-=e.w,this}subScalar(e){return this.x-=e,this.y-=e,this.z-=e,this.w-=e,this}subVectors(e,t){return this.x=e.x-t.x,this.y=e.y-t.y,this.z=e.z-t.z,this.w=e.w-t.w,this}multiply(e){return this.x*=e.x,this.y*=e.y,this.z*=e.z,this.w*=e.w,this}multiplyScalar(e){return this.x*=e,this.y*=e,this.z*=e,this.w*=e,this}applyMatrix4(e){const t=this.x,n=this.y,r=this.z,s=this.w,a=e.elements;return this.x=a[0]*t+a[4]*n+a[8]*r+a[12]*s,this.y=a[1]*t+a[5]*n+a[9]*r+a[13]*s,this.z=a[2]*t+a[6]*n+a[10]*r+a[14]*s,this.w=a[3]*t+a[7]*n+a[11]*r+a[15]*s,this}divide(e){return this.x/=e.x,this.y/=e.y,this.z/=e.z,this.w/=e.w,this}divideScalar(e){return this.multiplyScalar(1/e)}setAxisAngleFromQuaternion(e){this.w=2*Math.acos(e.w);const t=Math.sqrt(1-e.w*e.w);return t<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=e.x/t,this.y=e.y/t,this.z=e.z/t),this}setAxisAngleFromRotationMatrix(e){let t,n,r,s;const c=e.elements,l=c[0],u=c[4],d=c[8],h=c[1],f=c[5],_=c[9],y=c[2],g=c[6],m=c[10];if(Math.abs(u-h)<.01&&Math.abs(d-y)<.01&&Math.abs(_-g)<.01){if(Math.abs(u+h)<.1&&Math.abs(d+y)<.1&&Math.abs(_+g)<.1&&Math.abs(l+f+m-3)<.1)return this.set(1,0,0,0),this;t=Math.PI;const w=(l+1)/2,A=(f+1)/2,U=(m+1)/2,L=(u+h)/4,N=(d+y)/4,S=(_+g)/4;return w>A&&w>U?w<.01?(n=0,r=.707106781,s=.707106781):(n=Math.sqrt(w),r=L/n,s=N/n):A>U?A<.01?(n=.707106781,r=0,s=.707106781):(r=Math.sqrt(A),n=L/r,s=S/r):U<.01?(n=.707106781,r=.707106781,s=0):(s=Math.sqrt(U),n=N/s,r=S/s),this.set(n,r,s,t),this}let b=Math.sqrt((g-_)*(g-_)+(d-y)*(d-y)+(h-u)*(h-u));return Math.abs(b)<.001&&(b=1),this.x=(g-_)/b,this.y=(d-y)/b,this.z=(h-u)/b,this.w=Math.acos((l+f+m-1)/2),this}setFromMatrixPosition(e){const t=e.elements;return this.x=t[12],this.y=t[13],this.z=t[14],this.w=t[15],this}min(e){return this.x=Math.min(this.x,e.x),this.y=Math.min(this.y,e.y),this.z=Math.min(this.z,e.z),this.w=Math.min(this.w,e.w),this}max(e){return this.x=Math.max(this.x,e.x),this.y=Math.max(this.y,e.y),this.z=Math.max(this.z,e.z),this.w=Math.max(this.w,e.w),this}clamp(e,t){return this.x=nt(this.x,e.x,t.x),this.y=nt(this.y,e.y,t.y),this.z=nt(this.z,e.z,t.z),this.w=nt(this.w,e.w,t.w),this}clampScalar(e,t){return this.x=nt(this.x,e,t),this.y=nt(this.y,e,t),this.z=nt(this.z,e,t),this.w=nt(this.w,e,t),this}clampLength(e,t){const n=this.length();return this.divideScalar(n||1).multiplyScalar(nt(n,e,t))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z+this.w*e.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(e){return this.normalize().multiplyScalar(e)}lerp(e,t){return this.x+=(e.x-this.x)*t,this.y+=(e.y-this.y)*t,this.z+=(e.z-this.z)*t,this.w+=(e.w-this.w)*t,this}lerpVectors(e,t,n){return this.x=e.x+(t.x-e.x)*n,this.y=e.y+(t.y-e.y)*n,this.z=e.z+(t.z-e.z)*n,this.w=e.w+(t.w-e.w)*n,this}equals(e){return e.x===this.x&&e.y===this.y&&e.z===this.z&&e.w===this.w}fromArray(e,t=0){return this.x=e[t],this.y=e[t+1],this.z=e[t+2],this.w=e[t+3],this}toArray(e=[],t=0){return e[t]=this.x,e[t+1]=this.y,e[t+2]=this.z,e[t+3]=this.w,e}fromBufferAttribute(e,t){return this.x=e.getX(t),this.y=e.getY(t),this.z=e.getZ(t),this.w=e.getW(t),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}}class xh extends xi{constructor(e=1,t=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:bt,depthBuffer:!0,stencilBuffer:!1,resolveDepthBuffer:!0,resolveStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1},n),this.isRenderTarget=!0,this.width=e,this.height=t,this.depth=n.depth,this.scissor=new Et(0,0,e,t),this.scissorTest=!1,this.viewport=new Et(0,0,e,t),this.textures=[];const r={width:e,height:t,depth:n.depth},s=new Ht(r),a=n.count;for(let o=0;o<a;o++)this.textures[o]=s.clone(),this.textures[o].isRenderTargetTexture=!0,this.textures[o].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview}_setTextureOptions(e={}){const t={minFilter:bt,generateMipmaps:!1,flipY:!1,internalFormat:null};e.mapping!==void 0&&(t.mapping=e.mapping),e.wrapS!==void 0&&(t.wrapS=e.wrapS),e.wrapT!==void 0&&(t.wrapT=e.wrapT),e.wrapR!==void 0&&(t.wrapR=e.wrapR),e.magFilter!==void 0&&(t.magFilter=e.magFilter),e.minFilter!==void 0&&(t.minFilter=e.minFilter),e.format!==void 0&&(t.format=e.format),e.type!==void 0&&(t.type=e.type),e.anisotropy!==void 0&&(t.anisotropy=e.anisotropy),e.colorSpace!==void 0&&(t.colorSpace=e.colorSpace),e.flipY!==void 0&&(t.flipY=e.flipY),e.generateMipmaps!==void 0&&(t.generateMipmaps=e.generateMipmaps),e.internalFormat!==void 0&&(t.internalFormat=e.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(t)}get texture(){return this.textures[0]}set texture(e){this.textures[0]=e}set depthTexture(e){this._depthTexture!==null&&(this._depthTexture.renderTarget=null),e!==null&&(e.renderTarget=this),this._depthTexture=e}get depthTexture(){return this._depthTexture}setSize(e,t,n=1){if(this.width!==e||this.height!==t||this.depth!==n){this.width=e,this.height=t,this.depth=n;for(let r=0,s=this.textures.length;r<s;r++)this.textures[r].image.width=e,this.textures[r].image.height=t,this.textures[r].image.depth=n,this.textures[r].isData3DTexture!==!0&&(this.textures[r].isArrayTexture=this.textures[r].image.depth>1);this.dispose()}this.viewport.set(0,0,e,t),this.scissor.set(0,0,e,t)}clone(){return new this.constructor().copy(this)}copy(e){this.width=e.width,this.height=e.height,this.depth=e.depth,this.scissor.copy(e.scissor),this.scissorTest=e.scissorTest,this.viewport.copy(e.viewport),this.textures.length=0;for(let t=0,n=e.textures.length;t<n;t++){this.textures[t]=e.textures[t].clone(),this.textures[t].isRenderTargetTexture=!0,this.textures[t].renderTarget=this;const r=Object.assign({},e.textures[t].image);this.textures[t].source=new po(r)}return this.depthBuffer=e.depthBuffer,this.stencilBuffer=e.stencilBuffer,this.resolveDepthBuffer=e.resolveDepthBuffer,this.resolveStencilBuffer=e.resolveStencilBuffer,e.depthTexture!==null&&(this.depthTexture=e.depthTexture.clone()),this.samples=e.samples,this}dispose(){this.dispatchEvent({type:"dispose"})}}class wn extends xh{constructor(e=1,t=1,n={}){super(e,t,n),this.isWebGLRenderTarget=!0}}class ic extends Ht{constructor(e=null,t=1,n=1,r=1){super(null),this.isDataArrayTexture=!0,this.image={data:e,width:t,height:n,depth:r},this.magFilter=Nt,this.minFilter=Nt,this.wrapR=gn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}addLayerUpdate(e){this.layerUpdates.add(e)}clearLayerUpdates(){this.layerUpdates.clear()}}class vh extends Ht{constructor(e=null,t=1,n=1,r=1){super(null),this.isData3DTexture=!0,this.image={data:e,width:t,height:n,depth:r},this.magFilter=Nt,this.minFilter=Nt,this.wrapR=gn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class _t{constructor(e,t,n,r,s,a,o,c,l,u,d,h,f,_,y,g){_t.prototype.isMatrix4=!0,this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],e!==void 0&&this.set(e,t,n,r,s,a,o,c,l,u,d,h,f,_,y,g)}set(e,t,n,r,s,a,o,c,l,u,d,h,f,_,y,g){const m=this.elements;return m[0]=e,m[4]=t,m[8]=n,m[12]=r,m[1]=s,m[5]=a,m[9]=o,m[13]=c,m[2]=l,m[6]=u,m[10]=d,m[14]=h,m[3]=f,m[7]=_,m[11]=y,m[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new _t().fromArray(this.elements)}copy(e){const t=this.elements,n=e.elements;return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],this}copyPosition(e){const t=this.elements,n=e.elements;return t[12]=n[12],t[13]=n[13],t[14]=n[14],this}setFromMatrix3(e){const t=e.elements;return this.set(t[0],t[3],t[6],0,t[1],t[4],t[7],0,t[2],t[5],t[8],0,0,0,0,1),this}extractBasis(e,t,n){return this.determinant()===0?(e.set(1,0,0),t.set(0,1,0),n.set(0,0,1),this):(e.setFromMatrixColumn(this,0),t.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this)}makeBasis(e,t,n){return this.set(e.x,t.x,n.x,0,e.y,t.y,n.y,0,e.z,t.z,n.z,0,0,0,0,1),this}extractRotation(e){if(e.determinant()===0)return this.identity();const t=this.elements,n=e.elements,r=1/bi.setFromMatrixColumn(e,0).length(),s=1/bi.setFromMatrixColumn(e,1).length(),a=1/bi.setFromMatrixColumn(e,2).length();return t[0]=n[0]*r,t[1]=n[1]*r,t[2]=n[2]*r,t[3]=0,t[4]=n[4]*s,t[5]=n[5]*s,t[6]=n[6]*s,t[7]=0,t[8]=n[8]*a,t[9]=n[9]*a,t[10]=n[10]*a,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromEuler(e){const t=this.elements,n=e.x,r=e.y,s=e.z,a=Math.cos(n),o=Math.sin(n),c=Math.cos(r),l=Math.sin(r),u=Math.cos(s),d=Math.sin(s);if(e.order==="XYZ"){const h=a*u,f=a*d,_=o*u,y=o*d;t[0]=c*u,t[4]=-c*d,t[8]=l,t[1]=f+_*l,t[5]=h-y*l,t[9]=-o*c,t[2]=y-h*l,t[6]=_+f*l,t[10]=a*c}else if(e.order==="YXZ"){const h=c*u,f=c*d,_=l*u,y=l*d;t[0]=h+y*o,t[4]=_*o-f,t[8]=a*l,t[1]=a*d,t[5]=a*u,t[9]=-o,t[2]=f*o-_,t[6]=y+h*o,t[10]=a*c}else if(e.order==="ZXY"){const h=c*u,f=c*d,_=l*u,y=l*d;t[0]=h-y*o,t[4]=-a*d,t[8]=_+f*o,t[1]=f+_*o,t[5]=a*u,t[9]=y-h*o,t[2]=-a*l,t[6]=o,t[10]=a*c}else if(e.order==="ZYX"){const h=a*u,f=a*d,_=o*u,y=o*d;t[0]=c*u,t[4]=_*l-f,t[8]=h*l+y,t[1]=c*d,t[5]=y*l+h,t[9]=f*l-_,t[2]=-l,t[6]=o*c,t[10]=a*c}else if(e.order==="YZX"){const h=a*c,f=a*l,_=o*c,y=o*l;t[0]=c*u,t[4]=y-h*d,t[8]=_*d+f,t[1]=d,t[5]=a*u,t[9]=-o*u,t[2]=-l*u,t[6]=f*d+_,t[10]=h-y*d}else if(e.order==="XZY"){const h=a*c,f=a*l,_=o*c,y=o*l;t[0]=c*u,t[4]=-d,t[8]=l*u,t[1]=h*d+y,t[5]=a*u,t[9]=f*d-_,t[2]=_*d-f,t[6]=o*u,t[10]=y*d+h}return t[3]=0,t[7]=0,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,this}makeRotationFromQuaternion(e){return this.compose(Sh,e,Mh)}lookAt(e,t,n){const r=this.elements;return Zt.subVectors(e,t),Zt.lengthSq()===0&&(Zt.z=1),Zt.normalize(),Xn.crossVectors(n,Zt),Xn.lengthSq()===0&&(Math.abs(n.z)===1?Zt.x+=1e-4:Zt.z+=1e-4,Zt.normalize(),Xn.crossVectors(n,Zt)),Xn.normalize(),Cr.crossVectors(Zt,Xn),r[0]=Xn.x,r[4]=Cr.x,r[8]=Zt.x,r[1]=Xn.y,r[5]=Cr.y,r[9]=Zt.y,r[2]=Xn.z,r[6]=Cr.z,r[10]=Zt.z,this}multiply(e){return this.multiplyMatrices(this,e)}premultiply(e){return this.multiplyMatrices(e,this)}multiplyMatrices(e,t){const n=e.elements,r=t.elements,s=this.elements,a=n[0],o=n[4],c=n[8],l=n[12],u=n[1],d=n[5],h=n[9],f=n[13],_=n[2],y=n[6],g=n[10],m=n[14],b=n[3],w=n[7],A=n[11],U=n[15],L=r[0],N=r[4],S=r[8],T=r[12],G=r[1],D=r[5],O=r[9],V=r[13],K=r[2],Y=r[6],Z=r[10],X=r[14],fe=r[3],oe=r[7],ye=r[11],Ae=r[15];return s[0]=a*L+o*G+c*K+l*fe,s[4]=a*N+o*D+c*Y+l*oe,s[8]=a*S+o*O+c*Z+l*ye,s[12]=a*T+o*V+c*X+l*Ae,s[1]=u*L+d*G+h*K+f*fe,s[5]=u*N+d*D+h*Y+f*oe,s[9]=u*S+d*O+h*Z+f*ye,s[13]=u*T+d*V+h*X+f*Ae,s[2]=_*L+y*G+g*K+m*fe,s[6]=_*N+y*D+g*Y+m*oe,s[10]=_*S+y*O+g*Z+m*ye,s[14]=_*T+y*V+g*X+m*Ae,s[3]=b*L+w*G+A*K+U*fe,s[7]=b*N+w*D+A*Y+U*oe,s[11]=b*S+w*O+A*Z+U*ye,s[15]=b*T+w*V+A*X+U*Ae,this}multiplyScalar(e){const t=this.elements;return t[0]*=e,t[4]*=e,t[8]*=e,t[12]*=e,t[1]*=e,t[5]*=e,t[9]*=e,t[13]*=e,t[2]*=e,t[6]*=e,t[10]*=e,t[14]*=e,t[3]*=e,t[7]*=e,t[11]*=e,t[15]*=e,this}determinant(){const e=this.elements,t=e[0],n=e[4],r=e[8],s=e[12],a=e[1],o=e[5],c=e[9],l=e[13],u=e[2],d=e[6],h=e[10],f=e[14],_=e[3],y=e[7],g=e[11],m=e[15],b=c*f-l*h,w=o*f-l*d,A=o*h-c*d,U=a*f-l*u,L=a*h-c*u,N=a*d-o*u;return t*(y*b-g*w+m*A)-n*(_*b-g*U+m*L)+r*(_*w-y*U+m*N)-s*(_*A-y*L+g*N)}transpose(){const e=this.elements;let t;return t=e[1],e[1]=e[4],e[4]=t,t=e[2],e[2]=e[8],e[8]=t,t=e[6],e[6]=e[9],e[9]=t,t=e[3],e[3]=e[12],e[12]=t,t=e[7],e[7]=e[13],e[13]=t,t=e[11],e[11]=e[14],e[14]=t,this}setPosition(e,t,n){const r=this.elements;return e.isVector3?(r[12]=e.x,r[13]=e.y,r[14]=e.z):(r[12]=e,r[13]=t,r[14]=n),this}invert(){const e=this.elements,t=e[0],n=e[1],r=e[2],s=e[3],a=e[4],o=e[5],c=e[6],l=e[7],u=e[8],d=e[9],h=e[10],f=e[11],_=e[12],y=e[13],g=e[14],m=e[15],b=t*o-n*a,w=t*c-r*a,A=t*l-s*a,U=n*c-r*o,L=n*l-s*o,N=r*l-s*c,S=u*y-d*_,T=u*g-h*_,G=u*m-f*_,D=d*g-h*y,O=d*m-f*y,V=h*m-f*g,K=b*V-w*O+A*D+U*G-L*T+N*S;if(K===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);const Y=1/K;return e[0]=(o*V-c*O+l*D)*Y,e[1]=(r*O-n*V-s*D)*Y,e[2]=(y*N-g*L+m*U)*Y,e[3]=(h*L-d*N-f*U)*Y,e[4]=(c*G-a*V-l*T)*Y,e[5]=(t*V-r*G+s*T)*Y,e[6]=(g*A-_*N-m*w)*Y,e[7]=(u*N-h*A+f*w)*Y,e[8]=(a*O-o*G+l*S)*Y,e[9]=(n*G-t*O-s*S)*Y,e[10]=(_*L-y*A+m*b)*Y,e[11]=(d*A-u*L-f*b)*Y,e[12]=(o*T-a*D-c*S)*Y,e[13]=(t*D-n*T+r*S)*Y,e[14]=(y*w-_*U-g*b)*Y,e[15]=(u*U-d*w+h*b)*Y,this}scale(e){const t=this.elements,n=e.x,r=e.y,s=e.z;return t[0]*=n,t[4]*=r,t[8]*=s,t[1]*=n,t[5]*=r,t[9]*=s,t[2]*=n,t[6]*=r,t[10]*=s,t[3]*=n,t[7]*=r,t[11]*=s,this}getMaxScaleOnAxis(){const e=this.elements,t=e[0]*e[0]+e[1]*e[1]+e[2]*e[2],n=e[4]*e[4]+e[5]*e[5]+e[6]*e[6],r=e[8]*e[8]+e[9]*e[9]+e[10]*e[10];return Math.sqrt(Math.max(t,n,r))}makeTranslation(e,t,n){return e.isVector3?this.set(1,0,0,e.x,0,1,0,e.y,0,0,1,e.z,0,0,0,1):this.set(1,0,0,e,0,1,0,t,0,0,1,n,0,0,0,1),this}makeRotationX(e){const t=Math.cos(e),n=Math.sin(e);return this.set(1,0,0,0,0,t,-n,0,0,n,t,0,0,0,0,1),this}makeRotationY(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,0,n,0,0,1,0,0,-n,0,t,0,0,0,0,1),this}makeRotationZ(e){const t=Math.cos(e),n=Math.sin(e);return this.set(t,-n,0,0,n,t,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(e,t){const n=Math.cos(t),r=Math.sin(t),s=1-n,a=e.x,o=e.y,c=e.z,l=s*a,u=s*o;return this.set(l*a+n,l*o-r*c,l*c+r*o,0,l*o+r*c,u*o+n,u*c-r*a,0,l*c-r*o,u*c+r*a,s*c*c+n,0,0,0,0,1),this}makeScale(e,t,n){return this.set(e,0,0,0,0,t,0,0,0,0,n,0,0,0,0,1),this}makeShear(e,t,n,r,s,a){return this.set(1,n,s,0,e,1,a,0,t,r,1,0,0,0,0,1),this}compose(e,t,n){const r=this.elements,s=t._x,a=t._y,o=t._z,c=t._w,l=s+s,u=a+a,d=o+o,h=s*l,f=s*u,_=s*d,y=a*u,g=a*d,m=o*d,b=c*l,w=c*u,A=c*d,U=n.x,L=n.y,N=n.z;return r[0]=(1-(y+m))*U,r[1]=(f+A)*U,r[2]=(_-w)*U,r[3]=0,r[4]=(f-A)*L,r[5]=(1-(h+m))*L,r[6]=(g+b)*L,r[7]=0,r[8]=(_+w)*N,r[9]=(g-b)*N,r[10]=(1-(h+y))*N,r[11]=0,r[12]=e.x,r[13]=e.y,r[14]=e.z,r[15]=1,this}decompose(e,t,n){const r=this.elements;e.x=r[12],e.y=r[13],e.z=r[14];const s=this.determinant();if(s===0)return n.set(1,1,1),t.identity(),this;let a=bi.set(r[0],r[1],r[2]).length();const o=bi.set(r[4],r[5],r[6]).length(),c=bi.set(r[8],r[9],r[10]).length();s<0&&(a=-a),hn.copy(this);const l=1/a,u=1/o,d=1/c;return hn.elements[0]*=l,hn.elements[1]*=l,hn.elements[2]*=l,hn.elements[4]*=u,hn.elements[5]*=u,hn.elements[6]*=u,hn.elements[8]*=d,hn.elements[9]*=d,hn.elements[10]*=d,t.setFromRotationMatrix(hn),n.x=a,n.y=o,n.z=c,this}makePerspective(e,t,n,r,s,a,o=Tn,c=!1){const l=this.elements,u=2*s/(t-e),d=2*s/(n-r),h=(t+e)/(t-e),f=(n+r)/(n-r);let _,y;if(c)_=s/(a-s),y=a*s/(a-s);else if(o===Tn)_=-(a+s)/(a-s),y=-2*a*s/(a-s);else if(o===_r)_=-a/(a-s),y=-a*s/(a-s);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return l[0]=u,l[4]=0,l[8]=h,l[12]=0,l[1]=0,l[5]=d,l[9]=f,l[13]=0,l[2]=0,l[6]=0,l[10]=_,l[14]=y,l[3]=0,l[7]=0,l[11]=-1,l[15]=0,this}makeOrthographic(e,t,n,r,s,a,o=Tn,c=!1){const l=this.elements,u=2/(t-e),d=2/(n-r),h=-(t+e)/(t-e),f=-(n+r)/(n-r);let _,y;if(c)_=1/(a-s),y=a/(a-s);else if(o===Tn)_=-2/(a-s),y=-(a+s)/(a-s);else if(o===_r)_=-1/(a-s),y=-s/(a-s);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return l[0]=u,l[4]=0,l[8]=0,l[12]=h,l[1]=0,l[5]=d,l[9]=0,l[13]=f,l[2]=0,l[6]=0,l[10]=_,l[14]=y,l[3]=0,l[7]=0,l[11]=0,l[15]=1,this}equals(e){const t=this.elements,n=e.elements;for(let r=0;r<16;r++)if(t[r]!==n[r])return!1;return!0}fromArray(e,t=0){for(let n=0;n<16;n++)this.elements[n]=e[n+t];return this}toArray(e=[],t=0){const n=this.elements;return e[t]=n[0],e[t+1]=n[1],e[t+2]=n[2],e[t+3]=n[3],e[t+4]=n[4],e[t+5]=n[5],e[t+6]=n[6],e[t+7]=n[7],e[t+8]=n[8],e[t+9]=n[9],e[t+10]=n[10],e[t+11]=n[11],e[t+12]=n[12],e[t+13]=n[13],e[t+14]=n[14],e[t+15]=n[15],e}}const bi=new q,hn=new _t,Sh=new q(0,0,0),Mh=new q(1,1,1),Xn=new q,Cr=new q,Zt=new q,No=new _t,Oo=new si;class Rn{constructor(e=0,t=0,n=0,r=Rn.DEFAULT_ORDER){this.isEuler=!0,this._x=e,this._y=t,this._z=n,this._order=r}get x(){return this._x}set x(e){this._x=e,this._onChangeCallback()}get y(){return this._y}set y(e){this._y=e,this._onChangeCallback()}get z(){return this._z}set z(e){this._z=e,this._onChangeCallback()}get order(){return this._order}set order(e){this._order=e,this._onChangeCallback()}set(e,t,n,r=this._order){return this._x=e,this._y=t,this._z=n,this._order=r,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(e){return this._x=e._x,this._y=e._y,this._z=e._z,this._order=e._order,this._onChangeCallback(),this}setFromRotationMatrix(e,t=this._order,n=!0){const r=e.elements,s=r[0],a=r[4],o=r[8],c=r[1],l=r[5],u=r[9],d=r[2],h=r[6],f=r[10];switch(t){case"XYZ":this._y=Math.asin(nt(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-u,f),this._z=Math.atan2(-a,s)):(this._x=Math.atan2(h,l),this._z=0);break;case"YXZ":this._x=Math.asin(-nt(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(o,f),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-d,s),this._z=0);break;case"ZXY":this._x=Math.asin(nt(h,-1,1)),Math.abs(h)<.9999999?(this._y=Math.atan2(-d,f),this._z=Math.atan2(-a,l)):(this._y=0,this._z=Math.atan2(c,s));break;case"ZYX":this._y=Math.asin(-nt(d,-1,1)),Math.abs(d)<.9999999?(this._x=Math.atan2(h,f),this._z=Math.atan2(c,s)):(this._x=0,this._z=Math.atan2(-a,l));break;case"YZX":this._z=Math.asin(nt(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-u,l),this._y=Math.atan2(-d,s)):(this._x=0,this._y=Math.atan2(o,f));break;case"XZY":this._z=Math.asin(-nt(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(h,l),this._y=Math.atan2(o,s)):(this._x=Math.atan2(-u,f),this._y=0);break;default:Xe("Euler: .setFromRotationMatrix() encountered an unknown order: "+t)}return this._order=t,n===!0&&this._onChangeCallback(),this}setFromQuaternion(e,t,n){return No.makeRotationFromQuaternion(e),this.setFromRotationMatrix(No,t,n)}setFromVector3(e,t=this._order){return this.set(e.x,e.y,e.z,t)}reorder(e){return Oo.setFromEuler(this),this.setFromQuaternion(Oo,e)}equals(e){return e._x===this._x&&e._y===this._y&&e._z===this._z&&e._order===this._order}fromArray(e){return this._x=e[0],this._y=e[1],this._z=e[2],e[3]!==void 0&&(this._order=e[3]),this._onChangeCallback(),this}toArray(e=[],t=0){return e[t]=this._x,e[t+1]=this._y,e[t+2]=this._z,e[t+3]=this._order,e}_onChange(e){return this._onChangeCallback=e,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}}Rn.DEFAULT_ORDER="XYZ";class mo{constructor(){this.mask=1}set(e){this.mask=(1<<e|0)>>>0}enable(e){this.mask|=1<<e|0}enableAll(){this.mask=-1}toggle(e){this.mask^=1<<e|0}disable(e){this.mask&=~(1<<e|0)}disableAll(){this.mask=0}test(e){return(this.mask&e.mask)!==0}isEnabled(e){return(this.mask&(1<<e|0))!==0}}let yh=0;const Bo=new q,Ti=new si,Dn=new _t,Rr=new q,Qi=new q,Eh=new q,bh=new si,ko=new q(1,0,0),zo=new q(0,1,0),Go=new q(0,0,1),Ho={type:"added"},Th={type:"removed"},Ai={type:"childadded",child:null},Rs={type:"childremoved",child:null};class Ot extends xi{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:yh++}),this.uuid=vr(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=Ot.DEFAULT_UP.clone();const e=new q,t=new Rn,n=new si,r=new q(1,1,1);function s(){n.setFromEuler(t,!1)}function a(){t.setFromQuaternion(n,void 0,!1)}t._onChange(s),n._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:e},rotation:{configurable:!0,enumerable:!0,value:t},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:r},modelViewMatrix:{value:new _t},normalMatrix:{value:new Je}}),this.matrix=new _t,this.matrixWorld=new _t,this.matrixAutoUpdate=Ot.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=Ot.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new mo,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(e){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(e),this.matrix.decompose(this.position,this.quaternion,this.scale)}applyQuaternion(e){return this.quaternion.premultiply(e),this}setRotationFromAxisAngle(e,t){this.quaternion.setFromAxisAngle(e,t)}setRotationFromEuler(e){this.quaternion.setFromEuler(e,!0)}setRotationFromMatrix(e){this.quaternion.setFromRotationMatrix(e)}setRotationFromQuaternion(e){this.quaternion.copy(e)}rotateOnAxis(e,t){return Ti.setFromAxisAngle(e,t),this.quaternion.multiply(Ti),this}rotateOnWorldAxis(e,t){return Ti.setFromAxisAngle(e,t),this.quaternion.premultiply(Ti),this}rotateX(e){return this.rotateOnAxis(ko,e)}rotateY(e){return this.rotateOnAxis(zo,e)}rotateZ(e){return this.rotateOnAxis(Go,e)}translateOnAxis(e,t){return Bo.copy(e).applyQuaternion(this.quaternion),this.position.add(Bo.multiplyScalar(t)),this}translateX(e){return this.translateOnAxis(ko,e)}translateY(e){return this.translateOnAxis(zo,e)}translateZ(e){return this.translateOnAxis(Go,e)}localToWorld(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(this.matrixWorld)}worldToLocal(e){return this.updateWorldMatrix(!0,!1),e.applyMatrix4(Dn.copy(this.matrixWorld).invert())}lookAt(e,t,n){e.isVector3?Rr.copy(e):Rr.set(e,t,n);const r=this.parent;this.updateWorldMatrix(!0,!1),Qi.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Dn.lookAt(Qi,Rr,this.up):Dn.lookAt(Rr,Qi,this.up),this.quaternion.setFromRotationMatrix(Dn),r&&(Dn.extractRotation(r.matrixWorld),Ti.setFromRotationMatrix(Dn),this.quaternion.premultiply(Ti.invert()))}add(e){if(arguments.length>1){for(let t=0;t<arguments.length;t++)this.add(arguments[t]);return this}return e===this?(ot("Object3D.add: object can't be added as a child of itself.",e),this):(e&&e.isObject3D?(e.removeFromParent(),e.parent=this,this.children.push(e),e.dispatchEvent(Ho),Ai.child=e,this.dispatchEvent(Ai),Ai.child=null):ot("Object3D.add: object not an instance of THREE.Object3D.",e),this)}remove(e){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}const t=this.children.indexOf(e);return t!==-1&&(e.parent=null,this.children.splice(t,1),e.dispatchEvent(Th),Rs.child=e,this.dispatchEvent(Rs),Rs.child=null),this}removeFromParent(){const e=this.parent;return e!==null&&e.remove(this),this}clear(){return this.remove(...this.children)}attach(e){return this.updateWorldMatrix(!0,!1),Dn.copy(this.matrixWorld).invert(),e.parent!==null&&(e.parent.updateWorldMatrix(!0,!1),Dn.multiply(e.parent.matrixWorld)),e.applyMatrix4(Dn),e.removeFromParent(),e.parent=this,this.children.push(e),e.updateWorldMatrix(!1,!0),e.dispatchEvent(Ho),Ai.child=e,this.dispatchEvent(Ai),Ai.child=null,this}getObjectById(e){return this.getObjectByProperty("id",e)}getObjectByName(e){return this.getObjectByProperty("name",e)}getObjectByProperty(e,t){if(this[e]===t)return this;for(let n=0,r=this.children.length;n<r;n++){const a=this.children[n].getObjectByProperty(e,t);if(a!==void 0)return a}}getObjectsByProperty(e,t,n=[]){this[e]===t&&n.push(this);const r=this.children;for(let s=0,a=r.length;s<a;s++)r[s].getObjectsByProperty(e,t,n);return n}getWorldPosition(e){return this.updateWorldMatrix(!0,!1),e.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Qi,e,Eh),e}getWorldScale(e){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Qi,bh,e),e}getWorldDirection(e){this.updateWorldMatrix(!0,!1);const t=this.matrixWorld.elements;return e.set(t[8],t[9],t[10]).normalize()}raycast(){}traverse(e){e(this);const t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].traverse(e)}traverseVisible(e){if(this.visible===!1)return;e(this);const t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].traverseVisible(e)}traverseAncestors(e){const t=this.parent;t!==null&&(e(t),t.traverseAncestors(e))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);const e=this.pivot;if(e!==null){const t=e.x,n=e.y,r=e.z,s=this.matrix.elements;s[12]+=t-s[0]*t-s[4]*n-s[8]*r,s[13]+=n-s[1]*t-s[5]*n-s[9]*r,s[14]+=r-s[2]*t-s[6]*n-s[10]*r}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(e){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||e)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,e=!0);const t=this.children;for(let n=0,r=t.length;n<r;n++)t[n].updateMatrixWorld(e)}updateWorldMatrix(e,t){const n=this.parent;if(e===!0&&n!==null&&n.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),t===!0){const r=this.children;for(let s=0,a=r.length;s<a;s++)r[s].updateWorldMatrix(!1,!0)}}toJSON(e){const t=e===void 0||typeof e=="string",n={};t&&(e={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});const r={};r.uuid=this.uuid,r.type=this.type,this.name!==""&&(r.name=this.name),this.castShadow===!0&&(r.castShadow=!0),this.receiveShadow===!0&&(r.receiveShadow=!0),this.visible===!1&&(r.visible=!1),this.frustumCulled===!1&&(r.frustumCulled=!1),this.renderOrder!==0&&(r.renderOrder=this.renderOrder),this.static!==!1&&(r.static=this.static),Object.keys(this.userData).length>0&&(r.userData=this.userData),r.layers=this.layers.mask,r.matrix=this.matrix.toArray(),r.up=this.up.toArray(),this.pivot!==null&&(r.pivot=this.pivot.toArray()),this.matrixAutoUpdate===!1&&(r.matrixAutoUpdate=!1),this.morphTargetDictionary!==void 0&&(r.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(r.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(r.type="InstancedMesh",r.count=this.count,r.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(r.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(r.type="BatchedMesh",r.perObjectFrustumCulled=this.perObjectFrustumCulled,r.sortObjects=this.sortObjects,r.drawRanges=this._drawRanges,r.reservedRanges=this._reservedRanges,r.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),r.instanceInfo=this._instanceInfo.map(o=>({...o})),r.availableInstanceIds=this._availableInstanceIds.slice(),r.availableGeometryIds=this._availableGeometryIds.slice(),r.nextIndexStart=this._nextIndexStart,r.nextVertexStart=this._nextVertexStart,r.geometryCount=this._geometryCount,r.maxInstanceCount=this._maxInstanceCount,r.maxVertexCount=this._maxVertexCount,r.maxIndexCount=this._maxIndexCount,r.geometryInitialized=this._geometryInitialized,r.matricesTexture=this._matricesTexture.toJSON(e),r.indirectTexture=this._indirectTexture.toJSON(e),this._colorsTexture!==null&&(r.colorsTexture=this._colorsTexture.toJSON(e)),this.boundingSphere!==null&&(r.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(r.boundingBox=this.boundingBox.toJSON()));function s(o,c){return o[c.uuid]===void 0&&(o[c.uuid]=c.toJSON(e)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?r.background=this.background.toJSON():this.background.isTexture&&(r.background=this.background.toJSON(e).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(r.environment=this.environment.toJSON(e).uuid);else if(this.isMesh||this.isLine||this.isPoints){r.geometry=s(e.geometries,this.geometry);const o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){const c=o.shapes;if(Array.isArray(c))for(let l=0,u=c.length;l<u;l++){const d=c[l];s(e.shapes,d)}else s(e.shapes,c)}}if(this.isSkinnedMesh&&(r.bindMode=this.bindMode,r.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(s(e.skeletons,this.skeleton),r.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){const o=[];for(let c=0,l=this.material.length;c<l;c++)o.push(s(e.materials,this.material[c]));r.material=o}else r.material=s(e.materials,this.material);if(this.children.length>0){r.children=[];for(let o=0;o<this.children.length;o++)r.children.push(this.children[o].toJSON(e).object)}if(this.animations.length>0){r.animations=[];for(let o=0;o<this.animations.length;o++){const c=this.animations[o];r.animations.push(s(e.animations,c))}}if(t){const o=a(e.geometries),c=a(e.materials),l=a(e.textures),u=a(e.images),d=a(e.shapes),h=a(e.skeletons),f=a(e.animations),_=a(e.nodes);o.length>0&&(n.geometries=o),c.length>0&&(n.materials=c),l.length>0&&(n.textures=l),u.length>0&&(n.images=u),d.length>0&&(n.shapes=d),h.length>0&&(n.skeletons=h),f.length>0&&(n.animations=f),_.length>0&&(n.nodes=_)}return n.object=r,n;function a(o){const c=[];for(const l in o){const u=o[l];delete u.metadata,c.push(u)}return c}}clone(e){return new this.constructor().copy(this,e)}copy(e,t=!0){if(this.name=e.name,this.up.copy(e.up),this.position.copy(e.position),this.rotation.order=e.rotation.order,this.quaternion.copy(e.quaternion),this.scale.copy(e.scale),e.pivot!==null&&(this.pivot=e.pivot.clone()),this.matrix.copy(e.matrix),this.matrixWorld.copy(e.matrixWorld),this.matrixAutoUpdate=e.matrixAutoUpdate,this.matrixWorldAutoUpdate=e.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=e.matrixWorldNeedsUpdate,this.layers.mask=e.layers.mask,this.visible=e.visible,this.castShadow=e.castShadow,this.receiveShadow=e.receiveShadow,this.frustumCulled=e.frustumCulled,this.renderOrder=e.renderOrder,this.static=e.static,this.animations=e.animations.slice(),this.userData=JSON.parse(JSON.stringify(e.userData)),t===!0)for(let n=0;n<e.children.length;n++){const r=e.children[n];this.add(r.clone())}return this}}Ot.DEFAULT_UP=new q(0,1,0);Ot.DEFAULT_MATRIX_AUTO_UPDATE=!0;Ot.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;class Pr extends Ot{constructor(){super(),this.isGroup=!0,this.type="Group"}}const Ah={type:"move"};class Ps{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new Pr,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new Pr,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new q,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new q),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new Pr,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new q,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new q),this._grip}dispatchEvent(e){return this._targetRay!==null&&this._targetRay.dispatchEvent(e),this._grip!==null&&this._grip.dispatchEvent(e),this._hand!==null&&this._hand.dispatchEvent(e),this}connect(e){if(e&&e.hand){const t=this._hand;if(t)for(const n of e.hand.values())this._getHandJoint(t,n)}return this.dispatchEvent({type:"connected",data:e}),this}disconnect(e){return this.dispatchEvent({type:"disconnected",data:e}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(e,t,n){let r=null,s=null,a=null;const o=this._targetRay,c=this._grip,l=this._hand;if(e&&t.session.visibilityState!=="visible-blurred"){if(l&&e.hand){a=!0;for(const y of e.hand.values()){const g=t.getJointPose(y,n),m=this._getHandJoint(l,y);g!==null&&(m.matrix.fromArray(g.transform.matrix),m.matrix.decompose(m.position,m.rotation,m.scale),m.matrixWorldNeedsUpdate=!0,m.jointRadius=g.radius),m.visible=g!==null}const u=l.joints["index-finger-tip"],d=l.joints["thumb-tip"],h=u.position.distanceTo(d.position),f=.02,_=.005;l.inputState.pinching&&h>f+_?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:e.handedness,target:this})):!l.inputState.pinching&&h<=f-_&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:e.handedness,target:this}))}else c!==null&&e.gripSpace&&(s=t.getPose(e.gripSpace,n),s!==null&&(c.matrix.fromArray(s.transform.matrix),c.matrix.decompose(c.position,c.rotation,c.scale),c.matrixWorldNeedsUpdate=!0,s.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(s.linearVelocity)):c.hasLinearVelocity=!1,s.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(s.angularVelocity)):c.hasAngularVelocity=!1));o!==null&&(r=t.getPose(e.targetRaySpace,n),r===null&&s!==null&&(r=s),r!==null&&(o.matrix.fromArray(r.transform.matrix),o.matrix.decompose(o.position,o.rotation,o.scale),o.matrixWorldNeedsUpdate=!0,r.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(r.linearVelocity)):o.hasLinearVelocity=!1,r.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(r.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(Ah)))}return o!==null&&(o.visible=r!==null),c!==null&&(c.visible=s!==null),l!==null&&(l.visible=a!==null),this}_getHandJoint(e,t){if(e.joints[t.jointName]===void 0){const n=new Pr;n.matrixAutoUpdate=!1,n.visible=!1,e.joints[t.jointName]=n,e.add(n)}return e.joints[t.jointName]}}const rc={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},Yn={h:0,s:0,l:0},Dr={h:0,s:0,l:0};function Ds(i,e,t){return t<0&&(t+=1),t>1&&(t-=1),t<1/6?i+(e-i)*6*t:t<1/2?e:t<2/3?i+(e-i)*6*(2/3-t):i}class rt{constructor(e,t,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(e,t,n)}set(e,t,n){if(t===void 0&&n===void 0){const r=e;r&&r.isColor?this.copy(r):typeof r=="number"?this.setHex(r):typeof r=="string"&&this.setStyle(r)}else this.setRGB(e,t,n);return this}setScalar(e){return this.r=e,this.g=e,this.b=e,this}setHex(e,t=nn){return e=Math.floor(e),this.r=(e>>16&255)/255,this.g=(e>>8&255)/255,this.b=(e&255)/255,lt.colorSpaceToWorking(this,t),this}setRGB(e,t,n,r=lt.workingColorSpace){return this.r=e,this.g=t,this.b=n,lt.colorSpaceToWorking(this,r),this}setHSL(e,t,n,r=lt.workingColorSpace){if(e=dh(e,1),t=nt(t,0,1),n=nt(n,0,1),t===0)this.r=this.g=this.b=n;else{const s=n<=.5?n*(1+t):n+t-n*t,a=2*n-s;this.r=Ds(a,s,e+1/3),this.g=Ds(a,s,e),this.b=Ds(a,s,e-1/3)}return lt.colorSpaceToWorking(this,r),this}setStyle(e,t=nn){function n(s){s!==void 0&&parseFloat(s)<1&&Xe("Color: Alpha component of "+e+" will be ignored.")}let r;if(r=/^(\w+)\(([^\)]*)\)/.exec(e)){let s;const a=r[1],o=r[2];switch(a){case"rgb":case"rgba":if(s=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(s[4]),this.setRGB(Math.min(255,parseInt(s[1],10))/255,Math.min(255,parseInt(s[2],10))/255,Math.min(255,parseInt(s[3],10))/255,t);if(s=/^\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(s[4]),this.setRGB(Math.min(100,parseInt(s[1],10))/100,Math.min(100,parseInt(s[2],10))/100,Math.min(100,parseInt(s[3],10))/100,t);break;case"hsl":case"hsla":if(s=/^\s*(\d*\.?\d+)\s*,\s*(\d*\.?\d+)\%\s*,\s*(\d*\.?\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$/.exec(o))return n(s[4]),this.setHSL(parseFloat(s[1])/360,parseFloat(s[2])/100,parseFloat(s[3])/100,t);break;default:Xe("Color: Unknown color model "+e)}}else if(r=/^\#([A-Fa-f\d]+)$/.exec(e)){const s=r[1],a=s.length;if(a===3)return this.setRGB(parseInt(s.charAt(0),16)/15,parseInt(s.charAt(1),16)/15,parseInt(s.charAt(2),16)/15,t);if(a===6)return this.setHex(parseInt(s,16),t);Xe("Color: Invalid hex color "+e)}else if(e&&e.length>0)return this.setColorName(e,t);return this}setColorName(e,t=nn){const n=rc[e.toLowerCase()];return n!==void 0?this.setHex(n,t):Xe("Color: Unknown color "+e),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(e){return this.r=e.r,this.g=e.g,this.b=e.b,this}copySRGBToLinear(e){return this.r=Gn(e.r),this.g=Gn(e.g),this.b=Gn(e.b),this}copyLinearToSRGB(e){return this.r=Hi(e.r),this.g=Hi(e.g),this.b=Hi(e.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(e=nn){return lt.workingToColorSpace(kt.copy(this),e),Math.round(nt(kt.r*255,0,255))*65536+Math.round(nt(kt.g*255,0,255))*256+Math.round(nt(kt.b*255,0,255))}getHexString(e=nn){return("000000"+this.getHex(e).toString(16)).slice(-6)}getHSL(e,t=lt.workingColorSpace){lt.workingToColorSpace(kt.copy(this),t);const n=kt.r,r=kt.g,s=kt.b,a=Math.max(n,r,s),o=Math.min(n,r,s);let c,l;const u=(o+a)/2;if(o===a)c=0,l=0;else{const d=a-o;switch(l=u<=.5?d/(a+o):d/(2-a-o),a){case n:c=(r-s)/d+(r<s?6:0);break;case r:c=(s-n)/d+2;break;case s:c=(n-r)/d+4;break}c/=6}return e.h=c,e.s=l,e.l=u,e}getRGB(e,t=lt.workingColorSpace){return lt.workingToColorSpace(kt.copy(this),t),e.r=kt.r,e.g=kt.g,e.b=kt.b,e}getStyle(e=nn){lt.workingToColorSpace(kt.copy(this),e);const t=kt.r,n=kt.g,r=kt.b;return e!==nn?`color(${e} ${t.toFixed(3)} ${n.toFixed(3)} ${r.toFixed(3)})`:`rgb(${Math.round(t*255)},${Math.round(n*255)},${Math.round(r*255)})`}offsetHSL(e,t,n){return this.getHSL(Yn),this.setHSL(Yn.h+e,Yn.s+t,Yn.l+n)}add(e){return this.r+=e.r,this.g+=e.g,this.b+=e.b,this}addColors(e,t){return this.r=e.r+t.r,this.g=e.g+t.g,this.b=e.b+t.b,this}addScalar(e){return this.r+=e,this.g+=e,this.b+=e,this}sub(e){return this.r=Math.max(0,this.r-e.r),this.g=Math.max(0,this.g-e.g),this.b=Math.max(0,this.b-e.b),this}multiply(e){return this.r*=e.r,this.g*=e.g,this.b*=e.b,this}multiplyScalar(e){return this.r*=e,this.g*=e,this.b*=e,this}lerp(e,t){return this.r+=(e.r-this.r)*t,this.g+=(e.g-this.g)*t,this.b+=(e.b-this.b)*t,this}lerpColors(e,t,n){return this.r=e.r+(t.r-e.r)*n,this.g=e.g+(t.g-e.g)*n,this.b=e.b+(t.b-e.b)*n,this}lerpHSL(e,t){this.getHSL(Yn),e.getHSL(Dr);const n=bs(Yn.h,Dr.h,t),r=bs(Yn.s,Dr.s,t),s=bs(Yn.l,Dr.l,t);return this.setHSL(n,r,s),this}setFromVector3(e){return this.r=e.x,this.g=e.y,this.b=e.z,this}applyMatrix3(e){const t=this.r,n=this.g,r=this.b,s=e.elements;return this.r=s[0]*t+s[3]*n+s[6]*r,this.g=s[1]*t+s[4]*n+s[7]*r,this.b=s[2]*t+s[5]*n+s[8]*r,this}equals(e){return e.r===this.r&&e.g===this.g&&e.b===this.b}fromArray(e,t=0){return this.r=e[t],this.g=e[t+1],this.b=e[t+2],this}toArray(e=[],t=0){return e[t]=this.r,e[t+1]=this.g,e[t+2]=this.b,e}fromBufferAttribute(e,t){return this.r=e.getX(t),this.g=e.getY(t),this.b=e.getZ(t),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}}const kt=new rt;rt.NAMES=rc;class wh extends Ot{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new Rn,this.environmentIntensity=1,this.environmentRotation=new Rn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(e,t){return super.copy(e,t),e.background!==null&&(this.background=e.background.clone()),e.environment!==null&&(this.environment=e.environment.clone()),e.fog!==null&&(this.fog=e.fog.clone()),this.backgroundBlurriness=e.backgroundBlurriness,this.backgroundIntensity=e.backgroundIntensity,this.backgroundRotation.copy(e.backgroundRotation),this.environmentIntensity=e.environmentIntensity,this.environmentRotation.copy(e.environmentRotation),e.overrideMaterial!==null&&(this.overrideMaterial=e.overrideMaterial.clone()),this.matrixAutoUpdate=e.matrixAutoUpdate,this}toJSON(e){const t=super.toJSON(e);return this.fog!==null&&(t.object.fog=this.fog.toJSON()),this.backgroundBlurriness>0&&(t.object.backgroundBlurriness=this.backgroundBlurriness),this.backgroundIntensity!==1&&(t.object.backgroundIntensity=this.backgroundIntensity),t.object.backgroundRotation=this.backgroundRotation.toArray(),this.environmentIntensity!==1&&(t.object.environmentIntensity=this.environmentIntensity),t.object.environmentRotation=this.environmentRotation.toArray(),t}}const un=new q,In=new q,Is=new q,Ln=new q,wi=new q,Ci=new q,Vo=new q,Ls=new q,Us=new q,Fs=new q,Ns=new Et,Os=new Et,Bs=new Et;class sn{constructor(e=new q,t=new q,n=new q){this.a=e,this.b=t,this.c=n}static getNormal(e,t,n,r){r.subVectors(n,t),un.subVectors(e,t),r.cross(un);const s=r.lengthSq();return s>0?r.multiplyScalar(1/Math.sqrt(s)):r.set(0,0,0)}static getBarycoord(e,t,n,r,s){un.subVectors(r,t),In.subVectors(n,t),Is.subVectors(e,t);const a=un.dot(un),o=un.dot(In),c=un.dot(Is),l=In.dot(In),u=In.dot(Is),d=a*l-o*o;if(d===0)return s.set(0,0,0),null;const h=1/d,f=(l*c-o*u)*h,_=(a*u-o*c)*h;return s.set(1-f-_,_,f)}static containsPoint(e,t,n,r){return this.getBarycoord(e,t,n,r,Ln)===null?!1:Ln.x>=0&&Ln.y>=0&&Ln.x+Ln.y<=1}static getInterpolation(e,t,n,r,s,a,o,c){return this.getBarycoord(e,t,n,r,Ln)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(s,Ln.x),c.addScaledVector(a,Ln.y),c.addScaledVector(o,Ln.z),c)}static getInterpolatedAttribute(e,t,n,r,s,a){return Ns.setScalar(0),Os.setScalar(0),Bs.setScalar(0),Ns.fromBufferAttribute(e,t),Os.fromBufferAttribute(e,n),Bs.fromBufferAttribute(e,r),a.setScalar(0),a.addScaledVector(Ns,s.x),a.addScaledVector(Os,s.y),a.addScaledVector(Bs,s.z),a}static isFrontFacing(e,t,n,r){return un.subVectors(n,t),In.subVectors(e,t),un.cross(In).dot(r)<0}set(e,t,n){return this.a.copy(e),this.b.copy(t),this.c.copy(n),this}setFromPointsAndIndices(e,t,n,r){return this.a.copy(e[t]),this.b.copy(e[n]),this.c.copy(e[r]),this}setFromAttributeAndIndices(e,t,n,r){return this.a.fromBufferAttribute(e,t),this.b.fromBufferAttribute(e,n),this.c.fromBufferAttribute(e,r),this}clone(){return new this.constructor().copy(this)}copy(e){return this.a.copy(e.a),this.b.copy(e.b),this.c.copy(e.c),this}getArea(){return un.subVectors(this.c,this.b),In.subVectors(this.a,this.b),un.cross(In).length()*.5}getMidpoint(e){return e.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(e){return sn.getNormal(this.a,this.b,this.c,e)}getPlane(e){return e.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(e,t){return sn.getBarycoord(e,this.a,this.b,this.c,t)}getInterpolation(e,t,n,r,s){return sn.getInterpolation(e,this.a,this.b,this.c,t,n,r,s)}containsPoint(e){return sn.containsPoint(e,this.a,this.b,this.c)}isFrontFacing(e){return sn.isFrontFacing(this.a,this.b,this.c,e)}intersectsBox(e){return e.intersectsTriangle(this)}closestPointToPoint(e,t){const n=this.a,r=this.b,s=this.c;let a,o;wi.subVectors(r,n),Ci.subVectors(s,n),Ls.subVectors(e,n);const c=wi.dot(Ls),l=Ci.dot(Ls);if(c<=0&&l<=0)return t.copy(n);Us.subVectors(e,r);const u=wi.dot(Us),d=Ci.dot(Us);if(u>=0&&d<=u)return t.copy(r);const h=c*d-u*l;if(h<=0&&c>=0&&u<=0)return a=c/(c-u),t.copy(n).addScaledVector(wi,a);Fs.subVectors(e,s);const f=wi.dot(Fs),_=Ci.dot(Fs);if(_>=0&&f<=_)return t.copy(s);const y=f*l-c*_;if(y<=0&&l>=0&&_<=0)return o=l/(l-_),t.copy(n).addScaledVector(Ci,o);const g=u*_-f*d;if(g<=0&&d-u>=0&&f-_>=0)return Vo.subVectors(s,r),o=(d-u)/(d-u+(f-_)),t.copy(r).addScaledVector(Vo,o);const m=1/(g+y+h);return a=y*m,o=h*m,t.copy(n).addScaledVector(wi,a).addScaledVector(Ci,o)}equals(e){return e.a.equals(this.a)&&e.b.equals(this.b)&&e.c.equals(this.c)}}class vi{constructor(e=new q(1/0,1/0,1/0),t=new q(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=e,this.max=t}set(e,t){return this.min.copy(e),this.max.copy(t),this}setFromArray(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t+=3)this.expandByPoint(dn.fromArray(e,t));return this}setFromBufferAttribute(e){this.makeEmpty();for(let t=0,n=e.count;t<n;t++)this.expandByPoint(dn.fromBufferAttribute(e,t));return this}setFromPoints(e){this.makeEmpty();for(let t=0,n=e.length;t<n;t++)this.expandByPoint(e[t]);return this}setFromCenterAndSize(e,t){const n=dn.copy(t).multiplyScalar(.5);return this.min.copy(e).sub(n),this.max.copy(e).add(n),this}setFromObject(e,t=!1){return this.makeEmpty(),this.expandByObject(e,t)}clone(){return new this.constructor().copy(this)}copy(e){return this.min.copy(e.min),this.max.copy(e.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(e){return this.isEmpty()?e.set(0,0,0):e.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(e){return this.isEmpty()?e.set(0,0,0):e.subVectors(this.max,this.min)}expandByPoint(e){return this.min.min(e),this.max.max(e),this}expandByVector(e){return this.min.sub(e),this.max.add(e),this}expandByScalar(e){return this.min.addScalar(-e),this.max.addScalar(e),this}expandByObject(e,t=!1){e.updateWorldMatrix(!1,!1);const n=e.geometry;if(n!==void 0){const s=n.getAttribute("position");if(t===!0&&s!==void 0&&e.isInstancedMesh!==!0)for(let a=0,o=s.count;a<o;a++)e.isMesh===!0?e.getVertexPosition(a,dn):dn.fromBufferAttribute(s,a),dn.applyMatrix4(e.matrixWorld),this.expandByPoint(dn);else e.boundingBox!==void 0?(e.boundingBox===null&&e.computeBoundingBox(),Ir.copy(e.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),Ir.copy(n.boundingBox)),Ir.applyMatrix4(e.matrixWorld),this.union(Ir)}const r=e.children;for(let s=0,a=r.length;s<a;s++)this.expandByObject(r[s],t);return this}containsPoint(e){return e.x>=this.min.x&&e.x<=this.max.x&&e.y>=this.min.y&&e.y<=this.max.y&&e.z>=this.min.z&&e.z<=this.max.z}containsBox(e){return this.min.x<=e.min.x&&e.max.x<=this.max.x&&this.min.y<=e.min.y&&e.max.y<=this.max.y&&this.min.z<=e.min.z&&e.max.z<=this.max.z}getParameter(e,t){return t.set((e.x-this.min.x)/(this.max.x-this.min.x),(e.y-this.min.y)/(this.max.y-this.min.y),(e.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(e){return e.max.x>=this.min.x&&e.min.x<=this.max.x&&e.max.y>=this.min.y&&e.min.y<=this.max.y&&e.max.z>=this.min.z&&e.min.z<=this.max.z}intersectsSphere(e){return this.clampPoint(e.center,dn),dn.distanceToSquared(e.center)<=e.radius*e.radius}intersectsPlane(e){let t,n;return e.normal.x>0?(t=e.normal.x*this.min.x,n=e.normal.x*this.max.x):(t=e.normal.x*this.max.x,n=e.normal.x*this.min.x),e.normal.y>0?(t+=e.normal.y*this.min.y,n+=e.normal.y*this.max.y):(t+=e.normal.y*this.max.y,n+=e.normal.y*this.min.y),e.normal.z>0?(t+=e.normal.z*this.min.z,n+=e.normal.z*this.max.z):(t+=e.normal.z*this.max.z,n+=e.normal.z*this.min.z),t<=-e.constant&&n>=-e.constant}intersectsTriangle(e){if(this.isEmpty())return!1;this.getCenter(er),Lr.subVectors(this.max,er),Ri.subVectors(e.a,er),Pi.subVectors(e.b,er),Di.subVectors(e.c,er),qn.subVectors(Pi,Ri),Zn.subVectors(Di,Pi),oi.subVectors(Ri,Di);let t=[0,-qn.z,qn.y,0,-Zn.z,Zn.y,0,-oi.z,oi.y,qn.z,0,-qn.x,Zn.z,0,-Zn.x,oi.z,0,-oi.x,-qn.y,qn.x,0,-Zn.y,Zn.x,0,-oi.y,oi.x,0];return!ks(t,Ri,Pi,Di,Lr)||(t=[1,0,0,0,1,0,0,0,1],!ks(t,Ri,Pi,Di,Lr))?!1:(Ur.crossVectors(qn,Zn),t=[Ur.x,Ur.y,Ur.z],ks(t,Ri,Pi,Di,Lr))}clampPoint(e,t){return t.copy(e).clamp(this.min,this.max)}distanceToPoint(e){return this.clampPoint(e,dn).distanceTo(e)}getBoundingSphere(e){return this.isEmpty()?e.makeEmpty():(this.getCenter(e.center),e.radius=this.getSize(dn).length()*.5),e}intersect(e){return this.min.max(e.min),this.max.min(e.max),this.isEmpty()&&this.makeEmpty(),this}union(e){return this.min.min(e.min),this.max.max(e.max),this}applyMatrix4(e){return this.isEmpty()?this:(Un[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(e),Un[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(e),Un[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(e),Un[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(e),Un[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(e),Un[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(e),Un[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(e),Un[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(e),this.setFromPoints(Un),this)}translate(e){return this.min.add(e),this.max.add(e),this}equals(e){return e.min.equals(this.min)&&e.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(e){return this.min.fromArray(e.min),this.max.fromArray(e.max),this}}const Un=[new q,new q,new q,new q,new q,new q,new q,new q],dn=new q,Ir=new vi,Ri=new q,Pi=new q,Di=new q,qn=new q,Zn=new q,oi=new q,er=new q,Lr=new q,Ur=new q,li=new q;function ks(i,e,t,n,r){for(let s=0,a=i.length-3;s<=a;s+=3){li.fromArray(i,s);const o=r.x*Math.abs(li.x)+r.y*Math.abs(li.y)+r.z*Math.abs(li.z),c=e.dot(li),l=t.dot(li),u=n.dot(li);if(Math.max(-Math.max(c,l,u),Math.min(c,l,u))>o)return!1}return!0}const kn=Ch();function Ch(){const i=new ArrayBuffer(4),e=new Float32Array(i),t=new Uint32Array(i),n=new Uint32Array(512),r=new Uint32Array(512);for(let c=0;c<256;++c){const l=c-127;l<-27?(n[c]=0,n[c|256]=32768,r[c]=24,r[c|256]=24):l<-14?(n[c]=1024>>-l-14,n[c|256]=1024>>-l-14|32768,r[c]=-l-1,r[c|256]=-l-1):l<=15?(n[c]=l+15<<10,n[c|256]=l+15<<10|32768,r[c]=13,r[c|256]=13):l<128?(n[c]=31744,n[c|256]=64512,r[c]=24,r[c|256]=24):(n[c]=31744,n[c|256]=64512,r[c]=13,r[c|256]=13)}const s=new Uint32Array(2048),a=new Uint32Array(64),o=new Uint32Array(64);for(let c=1;c<1024;++c){let l=c<<13,u=0;for(;(l&8388608)===0;)l<<=1,u-=8388608;l&=-8388609,u+=947912704,s[c]=l|u}for(let c=1024;c<2048;++c)s[c]=939524096+(c-1024<<13);for(let c=1;c<31;++c)a[c]=c<<23;a[31]=1199570944,a[32]=2147483648;for(let c=33;c<63;++c)a[c]=2147483648+(c-32<<23);a[63]=3347054592;for(let c=1;c<64;++c)c!==32&&(o[c]=1024);return{floatView:e,uint32View:t,baseTable:n,shiftTable:r,mantissaTable:s,exponentTable:a,offsetTable:o}}function Rh(i){Math.abs(i)>65504&&Xe("DataUtils.toHalfFloat(): Value out of range."),i=nt(i,-65504,65504),kn.floatView[0]=i;const e=kn.uint32View[0],t=e>>23&511;return kn.baseTable[t]+((e&8388607)>>kn.shiftTable[t])}function Ph(i){const e=i>>10;return kn.uint32View[0]=kn.mantissaTable[kn.offsetTable[e]+(i&1023)]+kn.exponentTable[e],kn.floatView[0]}class Wo{static toHalfFloat(e){return Rh(e)}static fromHalfFloat(e){return Ph(e)}}const Tt=new q,Fr=new $e;let Dh=0;class on{constructor(e,t,n=!1){if(Array.isArray(e))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:Dh++}),this.name="",this.array=e,this.itemSize=t,this.count=e!==void 0?e.length/t:0,this.normalized=n,this.usage=Ro,this.updateRanges=[],this.gpuType=Yt,this.version=0}onUploadCallback(){}set needsUpdate(e){e===!0&&this.version++}setUsage(e){return this.usage=e,this}addUpdateRange(e,t){this.updateRanges.push({start:e,count:t})}clearUpdateRanges(){this.updateRanges.length=0}copy(e){return this.name=e.name,this.array=new e.array.constructor(e.array),this.itemSize=e.itemSize,this.count=e.count,this.normalized=e.normalized,this.usage=e.usage,this.gpuType=e.gpuType,this}copyAt(e,t,n){e*=this.itemSize,n*=t.itemSize;for(let r=0,s=this.itemSize;r<s;r++)this.array[e+r]=t.array[n+r];return this}copyArray(e){return this.array.set(e),this}applyMatrix3(e){if(this.itemSize===2)for(let t=0,n=this.count;t<n;t++)Fr.fromBufferAttribute(this,t),Fr.applyMatrix3(e),this.setXY(t,Fr.x,Fr.y);else if(this.itemSize===3)for(let t=0,n=this.count;t<n;t++)Tt.fromBufferAttribute(this,t),Tt.applyMatrix3(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}applyMatrix4(e){for(let t=0,n=this.count;t<n;t++)Tt.fromBufferAttribute(this,t),Tt.applyMatrix4(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}applyNormalMatrix(e){for(let t=0,n=this.count;t<n;t++)Tt.fromBufferAttribute(this,t),Tt.applyNormalMatrix(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}transformDirection(e){for(let t=0,n=this.count;t<n;t++)Tt.fromBufferAttribute(this,t),Tt.transformDirection(e),this.setXYZ(t,Tt.x,Tt.y,Tt.z);return this}set(e,t=0){return this.array.set(e,t),this}getComponent(e,t){let n=this.array[e*this.itemSize+t];return this.normalized&&(n=Ji(n,this.array)),n}setComponent(e,t,n){return this.normalized&&(n=Vt(n,this.array)),this.array[e*this.itemSize+t]=n,this}getX(e){let t=this.array[e*this.itemSize];return this.normalized&&(t=Ji(t,this.array)),t}setX(e,t){return this.normalized&&(t=Vt(t,this.array)),this.array[e*this.itemSize]=t,this}getY(e){let t=this.array[e*this.itemSize+1];return this.normalized&&(t=Ji(t,this.array)),t}setY(e,t){return this.normalized&&(t=Vt(t,this.array)),this.array[e*this.itemSize+1]=t,this}getZ(e){let t=this.array[e*this.itemSize+2];return this.normalized&&(t=Ji(t,this.array)),t}setZ(e,t){return this.normalized&&(t=Vt(t,this.array)),this.array[e*this.itemSize+2]=t,this}getW(e){let t=this.array[e*this.itemSize+3];return this.normalized&&(t=Ji(t,this.array)),t}setW(e,t){return this.normalized&&(t=Vt(t,this.array)),this.array[e*this.itemSize+3]=t,this}setXY(e,t,n){return e*=this.itemSize,this.normalized&&(t=Vt(t,this.array),n=Vt(n,this.array)),this.array[e+0]=t,this.array[e+1]=n,this}setXYZ(e,t,n,r){return e*=this.itemSize,this.normalized&&(t=Vt(t,this.array),n=Vt(n,this.array),r=Vt(r,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=r,this}setXYZW(e,t,n,r,s){return e*=this.itemSize,this.normalized&&(t=Vt(t,this.array),n=Vt(n,this.array),r=Vt(r,this.array),s=Vt(s,this.array)),this.array[e+0]=t,this.array[e+1]=n,this.array[e+2]=r,this.array[e+3]=s,this}onUpload(e){return this.onUploadCallback=e,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){const e={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return this.name!==""&&(e.name=this.name),this.usage!==Ro&&(e.usage=this.usage),e}}class sc extends on{constructor(e,t,n){super(new Uint16Array(e),t,n)}}class ac extends on{constructor(e,t,n){super(new Uint32Array(e),t,n)}}class It extends on{constructor(e,t,n){super(new Float32Array(e),t,n)}}const Ih=new vi,tr=new q,zs=new q;class Yi{constructor(e=new q,t=-1){this.isSphere=!0,this.center=e,this.radius=t}set(e,t){return this.center.copy(e),this.radius=t,this}setFromPoints(e,t){const n=this.center;t!==void 0?n.copy(t):Ih.setFromPoints(e).getCenter(n);let r=0;for(let s=0,a=e.length;s<a;s++)r=Math.max(r,n.distanceToSquared(e[s]));return this.radius=Math.sqrt(r),this}copy(e){return this.center.copy(e.center),this.radius=e.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(e){return e.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(e){return e.distanceTo(this.center)-this.radius}intersectsSphere(e){const t=this.radius+e.radius;return e.center.distanceToSquared(this.center)<=t*t}intersectsBox(e){return e.intersectsSphere(this)}intersectsPlane(e){return Math.abs(e.distanceToPoint(this.center))<=this.radius}clampPoint(e,t){const n=this.center.distanceToSquared(e);return t.copy(e),n>this.radius*this.radius&&(t.sub(this.center).normalize(),t.multiplyScalar(this.radius).add(this.center)),t}getBoundingBox(e){return this.isEmpty()?(e.makeEmpty(),e):(e.set(this.center,this.center),e.expandByScalar(this.radius),e)}applyMatrix4(e){return this.center.applyMatrix4(e),this.radius=this.radius*e.getMaxScaleOnAxis(),this}translate(e){return this.center.add(e),this}expandByPoint(e){if(this.isEmpty())return this.center.copy(e),this.radius=0,this;tr.subVectors(e,this.center);const t=tr.lengthSq();if(t>this.radius*this.radius){const n=Math.sqrt(t),r=(n-this.radius)*.5;this.center.addScaledVector(tr,r/n),this.radius+=r}return this}union(e){return e.isEmpty()?this:this.isEmpty()?(this.copy(e),this):(this.center.equals(e.center)===!0?this.radius=Math.max(this.radius,e.radius):(zs.subVectors(e.center,this.center).setLength(e.radius),this.expandByPoint(tr.copy(e.center).add(zs)),this.expandByPoint(tr.copy(e.center).sub(zs))),this)}equals(e){return e.center.equals(this.center)&&e.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(e){return this.radius=e.radius,this.center.fromArray(e.center),this}}let Lh=0;const tn=new _t,Gs=new Ot,Ii=new q,$t=new vi,nr=new vi,Dt=new q;class Qt extends xi{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:Lh++}),this.uuid=vr(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={}}getIndex(){return this.index}setIndex(e){return Array.isArray(e)?this.index=new(lh(e)?ac:sc)(e,1):this.index=e,this}setIndirect(e,t=0){return this.indirect=e,this.indirectOffset=t,this}getIndirect(){return this.indirect}getAttribute(e){return this.attributes[e]}setAttribute(e,t){return this.attributes[e]=t,this}deleteAttribute(e){return delete this.attributes[e],this}hasAttribute(e){return this.attributes[e]!==void 0}addGroup(e,t,n=0){this.groups.push({start:e,count:t,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(e,t){this.drawRange.start=e,this.drawRange.count=t}applyMatrix4(e){const t=this.attributes.position;t!==void 0&&(t.applyMatrix4(e),t.needsUpdate=!0);const n=this.attributes.normal;if(n!==void 0){const s=new Je().getNormalMatrix(e);n.applyNormalMatrix(s),n.needsUpdate=!0}const r=this.attributes.tangent;return r!==void 0&&(r.transformDirection(e),r.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this}applyQuaternion(e){return tn.makeRotationFromQuaternion(e),this.applyMatrix4(tn),this}rotateX(e){return tn.makeRotationX(e),this.applyMatrix4(tn),this}rotateY(e){return tn.makeRotationY(e),this.applyMatrix4(tn),this}rotateZ(e){return tn.makeRotationZ(e),this.applyMatrix4(tn),this}translate(e,t,n){return tn.makeTranslation(e,t,n),this.applyMatrix4(tn),this}scale(e,t,n){return tn.makeScale(e,t,n),this.applyMatrix4(tn),this}lookAt(e){return Gs.lookAt(e),Gs.updateMatrix(),this.applyMatrix4(Gs.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(Ii).negate(),this.translate(Ii.x,Ii.y,Ii.z),this}setFromPoints(e){const t=this.getAttribute("position");if(t===void 0){const n=[];for(let r=0,s=e.length;r<s;r++){const a=e[r];n.push(a.x,a.y,a.z||0)}this.setAttribute("position",new It(n,3))}else{const n=Math.min(e.length,t.count);for(let r=0;r<n;r++){const s=e[r];t.setXYZ(r,s.x,s.y,s.z||0)}e.length>t.count&&Xe("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),t.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new vi);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){ot("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new q(-1/0,-1/0,-1/0),new q(1/0,1/0,1/0));return}if(e!==void 0){if(this.boundingBox.setFromBufferAttribute(e),t)for(let n=0,r=t.length;n<r;n++){const s=t[n];$t.setFromBufferAttribute(s),this.morphTargetsRelative?(Dt.addVectors(this.boundingBox.min,$t.min),this.boundingBox.expandByPoint(Dt),Dt.addVectors(this.boundingBox.max,$t.max),this.boundingBox.expandByPoint(Dt)):(this.boundingBox.expandByPoint($t.min),this.boundingBox.expandByPoint($t.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&ot('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Yi);const e=this.attributes.position,t=this.morphAttributes.position;if(e&&e.isGLBufferAttribute){ot("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new q,1/0);return}if(e){const n=this.boundingSphere.center;if($t.setFromBufferAttribute(e),t)for(let s=0,a=t.length;s<a;s++){const o=t[s];nr.setFromBufferAttribute(o),this.morphTargetsRelative?(Dt.addVectors($t.min,nr.min),$t.expandByPoint(Dt),Dt.addVectors($t.max,nr.max),$t.expandByPoint(Dt)):($t.expandByPoint(nr.min),$t.expandByPoint(nr.max))}$t.getCenter(n);let r=0;for(let s=0,a=e.count;s<a;s++)Dt.fromBufferAttribute(e,s),r=Math.max(r,n.distanceToSquared(Dt));if(t)for(let s=0,a=t.length;s<a;s++){const o=t[s],c=this.morphTargetsRelative;for(let l=0,u=o.count;l<u;l++)Dt.fromBufferAttribute(o,l),c&&(Ii.fromBufferAttribute(e,l),Dt.add(Ii)),r=Math.max(r,n.distanceToSquared(Dt))}this.boundingSphere.radius=Math.sqrt(r),isNaN(this.boundingSphere.radius)&&ot('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){const e=this.index,t=this.attributes;if(e===null||t.position===void 0||t.normal===void 0||t.uv===void 0){ot("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}const n=t.position,r=t.normal,s=t.uv;this.hasAttribute("tangent")===!1&&this.setAttribute("tangent",new on(new Float32Array(4*n.count),4));const a=this.getAttribute("tangent"),o=[],c=[];for(let S=0;S<n.count;S++)o[S]=new q,c[S]=new q;const l=new q,u=new q,d=new q,h=new $e,f=new $e,_=new $e,y=new q,g=new q;function m(S,T,G){l.fromBufferAttribute(n,S),u.fromBufferAttribute(n,T),d.fromBufferAttribute(n,G),h.fromBufferAttribute(s,S),f.fromBufferAttribute(s,T),_.fromBufferAttribute(s,G),u.sub(l),d.sub(l),f.sub(h),_.sub(h);const D=1/(f.x*_.y-_.x*f.y);isFinite(D)&&(y.copy(u).multiplyScalar(_.y).addScaledVector(d,-f.y).multiplyScalar(D),g.copy(d).multiplyScalar(f.x).addScaledVector(u,-_.x).multiplyScalar(D),o[S].add(y),o[T].add(y),o[G].add(y),c[S].add(g),c[T].add(g),c[G].add(g))}let b=this.groups;b.length===0&&(b=[{start:0,count:e.count}]);for(let S=0,T=b.length;S<T;++S){const G=b[S],D=G.start,O=G.count;for(let V=D,K=D+O;V<K;V+=3)m(e.getX(V+0),e.getX(V+1),e.getX(V+2))}const w=new q,A=new q,U=new q,L=new q;function N(S){U.fromBufferAttribute(r,S),L.copy(U);const T=o[S];w.copy(T),w.sub(U.multiplyScalar(U.dot(T))).normalize(),A.crossVectors(L,T);const D=A.dot(c[S])<0?-1:1;a.setXYZW(S,w.x,w.y,w.z,D)}for(let S=0,T=b.length;S<T;++S){const G=b[S],D=G.start,O=G.count;for(let V=D,K=D+O;V<K;V+=3)N(e.getX(V+0)),N(e.getX(V+1)),N(e.getX(V+2))}}computeVertexNormals(){const e=this.index,t=this.getAttribute("position");if(t!==void 0){let n=this.getAttribute("normal");if(n===void 0)n=new on(new Float32Array(t.count*3),3),this.setAttribute("normal",n);else for(let h=0,f=n.count;h<f;h++)n.setXYZ(h,0,0,0);const r=new q,s=new q,a=new q,o=new q,c=new q,l=new q,u=new q,d=new q;if(e)for(let h=0,f=e.count;h<f;h+=3){const _=e.getX(h+0),y=e.getX(h+1),g=e.getX(h+2);r.fromBufferAttribute(t,_),s.fromBufferAttribute(t,y),a.fromBufferAttribute(t,g),u.subVectors(a,s),d.subVectors(r,s),u.cross(d),o.fromBufferAttribute(n,_),c.fromBufferAttribute(n,y),l.fromBufferAttribute(n,g),o.add(u),c.add(u),l.add(u),n.setXYZ(_,o.x,o.y,o.z),n.setXYZ(y,c.x,c.y,c.z),n.setXYZ(g,l.x,l.y,l.z)}else for(let h=0,f=t.count;h<f;h+=3)r.fromBufferAttribute(t,h+0),s.fromBufferAttribute(t,h+1),a.fromBufferAttribute(t,h+2),u.subVectors(a,s),d.subVectors(r,s),u.cross(d),n.setXYZ(h+0,u.x,u.y,u.z),n.setXYZ(h+1,u.x,u.y,u.z),n.setXYZ(h+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){const e=this.attributes.normal;for(let t=0,n=e.count;t<n;t++)Dt.fromBufferAttribute(e,t),Dt.normalize(),e.setXYZ(t,Dt.x,Dt.y,Dt.z)}toNonIndexed(){function e(o,c){const l=o.array,u=o.itemSize,d=o.normalized,h=new l.constructor(c.length*u);let f=0,_=0;for(let y=0,g=c.length;y<g;y++){o.isInterleavedBufferAttribute?f=c[y]*o.data.stride+o.offset:f=c[y]*u;for(let m=0;m<u;m++)h[_++]=l[f++]}return new on(h,u,d)}if(this.index===null)return Xe("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;const t=new Qt,n=this.index.array,r=this.attributes;for(const o in r){const c=r[o],l=e(c,n);t.setAttribute(o,l)}const s=this.morphAttributes;for(const o in s){const c=[],l=s[o];for(let u=0,d=l.length;u<d;u++){const h=l[u],f=e(h,n);c.push(f)}t.morphAttributes[o]=c}t.morphTargetsRelative=this.morphTargetsRelative;const a=this.groups;for(let o=0,c=a.length;o<c;o++){const l=a[o];t.addGroup(l.start,l.count,l.materialIndex)}return t}toJSON(){const e={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(e.uuid=this.uuid,e.type=this.type,this.name!==""&&(e.name=this.name),Object.keys(this.userData).length>0&&(e.userData=this.userData),this.parameters!==void 0){const c=this.parameters;for(const l in c)c[l]!==void 0&&(e[l]=c[l]);return e}e.data={attributes:{}};const t=this.index;t!==null&&(e.data.index={type:t.array.constructor.name,array:Array.prototype.slice.call(t.array)});const n=this.attributes;for(const c in n){const l=n[c];e.data.attributes[c]=l.toJSON(e.data)}const r={};let s=!1;for(const c in this.morphAttributes){const l=this.morphAttributes[c],u=[];for(let d=0,h=l.length;d<h;d++){const f=l[d];u.push(f.toJSON(e.data))}u.length>0&&(r[c]=u,s=!0)}s&&(e.data.morphAttributes=r,e.data.morphTargetsRelative=this.morphTargetsRelative);const a=this.groups;a.length>0&&(e.data.groups=JSON.parse(JSON.stringify(a)));const o=this.boundingSphere;return o!==null&&(e.data.boundingSphere=o.toJSON()),e}clone(){return new this.constructor().copy(this)}copy(e){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;const t={};this.name=e.name;const n=e.index;n!==null&&this.setIndex(n.clone());const r=e.attributes;for(const l in r){const u=r[l];this.setAttribute(l,u.clone(t))}const s=e.morphAttributes;for(const l in s){const u=[],d=s[l];for(let h=0,f=d.length;h<f;h++)u.push(d[h].clone(t));this.morphAttributes[l]=u}this.morphTargetsRelative=e.morphTargetsRelative;const a=e.groups;for(let l=0,u=a.length;l<u;l++){const d=a[l];this.addGroup(d.start,d.count,d.materialIndex)}const o=e.boundingBox;o!==null&&(this.boundingBox=o.clone());const c=e.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=e.drawRange.start,this.drawRange.count=e.drawRange.count,this.userData=e.userData,this}dispose(){this.dispatchEvent({type:"dispose"})}}let Uh=0;class qi extends xi{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Uh++}),this.uuid=vr(),this.name="",this.type="Material",this.blending=Gi,this.side=ri,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=oa,this.blendDst=la,this.blendEquation=fi,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new rt(0,0,0),this.blendAlpha=0,this.depthFunc=Vi,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=Co,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=yi,this.stencilZFail=yi,this.stencilZPass=yi,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(e){this._alphaTest>0!=e>0&&this.version++,this._alphaTest=e}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(e){if(e!==void 0)for(const t in e){const n=e[t];if(n===void 0){Xe(`Material: parameter '${t}' has value of undefined.`);continue}const r=this[t];if(r===void 0){Xe(`Material: '${t}' is not a property of THREE.${this.type}.`);continue}r&&r.isColor?r.set(n):r&&r.isVector3&&n&&n.isVector3?r.copy(n):this[t]=n}}toJSON(e){const t=e===void 0||typeof e=="string";t&&(e={textures:{},images:{}});const n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,this.name!==""&&(n.name=this.name),this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&this.emissiveIntensity!==1&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(e).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(e).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(e).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(e).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(e).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessRange!==void 0&&(n.iridescenceThicknessRange=this.iridescenceThicknessRange),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(e).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(e).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(e).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(e).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(e).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(e).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(e).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(e).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(e).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(e).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(e).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(e).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(e).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(e).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(e).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(e).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(e).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(e).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(e).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(e).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(e).uuid),this.attenuationDistance!==void 0&&this.attenuationDistance!==1/0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.shadowSide!==null&&(n.shadowSide=this.shadowSide),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),this.blending!==Gi&&(n.blending=this.blending),this.side!==ri&&(n.side=this.side),this.vertexColors===!0&&(n.vertexColors=!0),this.opacity<1&&(n.opacity=this.opacity),this.transparent===!0&&(n.transparent=!0),this.blendSrc!==oa&&(n.blendSrc=this.blendSrc),this.blendDst!==la&&(n.blendDst=this.blendDst),this.blendEquation!==fi&&(n.blendEquation=this.blendEquation),this.blendSrcAlpha!==null&&(n.blendSrcAlpha=this.blendSrcAlpha),this.blendDstAlpha!==null&&(n.blendDstAlpha=this.blendDstAlpha),this.blendEquationAlpha!==null&&(n.blendEquationAlpha=this.blendEquationAlpha),this.blendColor&&this.blendColor.isColor&&(n.blendColor=this.blendColor.getHex()),this.blendAlpha!==0&&(n.blendAlpha=this.blendAlpha),this.depthFunc!==Vi&&(n.depthFunc=this.depthFunc),this.depthTest===!1&&(n.depthTest=this.depthTest),this.depthWrite===!1&&(n.depthWrite=this.depthWrite),this.colorWrite===!1&&(n.colorWrite=this.colorWrite),this.stencilWriteMask!==255&&(n.stencilWriteMask=this.stencilWriteMask),this.stencilFunc!==Co&&(n.stencilFunc=this.stencilFunc),this.stencilRef!==0&&(n.stencilRef=this.stencilRef),this.stencilFuncMask!==255&&(n.stencilFuncMask=this.stencilFuncMask),this.stencilFail!==yi&&(n.stencilFail=this.stencilFail),this.stencilZFail!==yi&&(n.stencilZFail=this.stencilZFail),this.stencilZPass!==yi&&(n.stencilZPass=this.stencilZPass),this.stencilWrite===!0&&(n.stencilWrite=this.stencilWrite),this.rotation!==void 0&&this.rotation!==0&&(n.rotation=this.rotation),this.polygonOffset===!0&&(n.polygonOffset=!0),this.polygonOffsetFactor!==0&&(n.polygonOffsetFactor=this.polygonOffsetFactor),this.polygonOffsetUnits!==0&&(n.polygonOffsetUnits=this.polygonOffsetUnits),this.linewidth!==void 0&&this.linewidth!==1&&(n.linewidth=this.linewidth),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.dithering===!0&&(n.dithering=!0),this.alphaTest>0&&(n.alphaTest=this.alphaTest),this.alphaHash===!0&&(n.alphaHash=!0),this.alphaToCoverage===!0&&(n.alphaToCoverage=!0),this.premultipliedAlpha===!0&&(n.premultipliedAlpha=!0),this.forceSinglePass===!0&&(n.forceSinglePass=!0),this.allowOverride===!1&&(n.allowOverride=!1),this.wireframe===!0&&(n.wireframe=!0),this.wireframeLinewidth>1&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!=="round"&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!=="round"&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading===!0&&(n.flatShading=!0),this.visible===!1&&(n.visible=!1),this.toneMapped===!1&&(n.toneMapped=!1),this.fog===!1&&(n.fog=!1),Object.keys(this.userData).length>0&&(n.userData=this.userData);function r(s){const a=[];for(const o in s){const c=s[o];delete c.metadata,a.push(c)}return a}if(t){const s=r(e.textures),a=r(e.images);s.length>0&&(n.textures=s),a.length>0&&(n.images=a)}return n}clone(){return new this.constructor().copy(this)}copy(e){this.name=e.name,this.blending=e.blending,this.side=e.side,this.vertexColors=e.vertexColors,this.opacity=e.opacity,this.transparent=e.transparent,this.blendSrc=e.blendSrc,this.blendDst=e.blendDst,this.blendEquation=e.blendEquation,this.blendSrcAlpha=e.blendSrcAlpha,this.blendDstAlpha=e.blendDstAlpha,this.blendEquationAlpha=e.blendEquationAlpha,this.blendColor.copy(e.blendColor),this.blendAlpha=e.blendAlpha,this.depthFunc=e.depthFunc,this.depthTest=e.depthTest,this.depthWrite=e.depthWrite,this.stencilWriteMask=e.stencilWriteMask,this.stencilFunc=e.stencilFunc,this.stencilRef=e.stencilRef,this.stencilFuncMask=e.stencilFuncMask,this.stencilFail=e.stencilFail,this.stencilZFail=e.stencilZFail,this.stencilZPass=e.stencilZPass,this.stencilWrite=e.stencilWrite;const t=e.clippingPlanes;let n=null;if(t!==null){const r=t.length;n=new Array(r);for(let s=0;s!==r;++s)n[s]=t[s].clone()}return this.clippingPlanes=n,this.clipIntersection=e.clipIntersection,this.clipShadows=e.clipShadows,this.shadowSide=e.shadowSide,this.colorWrite=e.colorWrite,this.precision=e.precision,this.polygonOffset=e.polygonOffset,this.polygonOffsetFactor=e.polygonOffsetFactor,this.polygonOffsetUnits=e.polygonOffsetUnits,this.dithering=e.dithering,this.alphaTest=e.alphaTest,this.alphaHash=e.alphaHash,this.alphaToCoverage=e.alphaToCoverage,this.premultipliedAlpha=e.premultipliedAlpha,this.forceSinglePass=e.forceSinglePass,this.allowOverride=e.allowOverride,this.visible=e.visible,this.toneMapped=e.toneMapped,this.userData=JSON.parse(JSON.stringify(e.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(e){e===!0&&this.version++}}const Fn=new q,Hs=new q,Nr=new q,$n=new q,Vs=new q,Or=new q,Ws=new q;class _s{constructor(e=new q,t=new q(0,0,-1)){this.origin=e,this.direction=t}set(e,t){return this.origin.copy(e),this.direction.copy(t),this}copy(e){return this.origin.copy(e.origin),this.direction.copy(e.direction),this}at(e,t){return t.copy(this.origin).addScaledVector(this.direction,e)}lookAt(e){return this.direction.copy(e).sub(this.origin).normalize(),this}recast(e){return this.origin.copy(this.at(e,Fn)),this}closestPointToPoint(e,t){t.subVectors(e,this.origin);const n=t.dot(this.direction);return n<0?t.copy(this.origin):t.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(e){return Math.sqrt(this.distanceSqToPoint(e))}distanceSqToPoint(e){const t=Fn.subVectors(e,this.origin).dot(this.direction);return t<0?this.origin.distanceToSquared(e):(Fn.copy(this.origin).addScaledVector(this.direction,t),Fn.distanceToSquared(e))}distanceSqToSegment(e,t,n,r){Hs.copy(e).add(t).multiplyScalar(.5),Nr.copy(t).sub(e).normalize(),$n.copy(this.origin).sub(Hs);const s=e.distanceTo(t)*.5,a=-this.direction.dot(Nr),o=$n.dot(this.direction),c=-$n.dot(Nr),l=$n.lengthSq(),u=Math.abs(1-a*a);let d,h,f,_;if(u>0)if(d=a*c-o,h=a*o-c,_=s*u,d>=0)if(h>=-_)if(h<=_){const y=1/u;d*=y,h*=y,f=d*(d+a*h+2*o)+h*(a*d+h+2*c)+l}else h=s,d=Math.max(0,-(a*h+o)),f=-d*d+h*(h+2*c)+l;else h=-s,d=Math.max(0,-(a*h+o)),f=-d*d+h*(h+2*c)+l;else h<=-_?(d=Math.max(0,-(-a*s+o)),h=d>0?-s:Math.min(Math.max(-s,-c),s),f=-d*d+h*(h+2*c)+l):h<=_?(d=0,h=Math.min(Math.max(-s,-c),s),f=h*(h+2*c)+l):(d=Math.max(0,-(a*s+o)),h=d>0?s:Math.min(Math.max(-s,-c),s),f=-d*d+h*(h+2*c)+l);else h=a>0?-s:s,d=Math.max(0,-(a*h+o)),f=-d*d+h*(h+2*c)+l;return n&&n.copy(this.origin).addScaledVector(this.direction,d),r&&r.copy(Hs).addScaledVector(Nr,h),f}intersectSphere(e,t){Fn.subVectors(e.center,this.origin);const n=Fn.dot(this.direction),r=Fn.dot(Fn)-n*n,s=e.radius*e.radius;if(r>s)return null;const a=Math.sqrt(s-r),o=n-a,c=n+a;return c<0?null:o<0?this.at(c,t):this.at(o,t)}intersectsSphere(e){return e.radius<0?!1:this.distanceSqToPoint(e.center)<=e.radius*e.radius}distanceToPlane(e){const t=e.normal.dot(this.direction);if(t===0)return e.distanceToPoint(this.origin)===0?0:null;const n=-(this.origin.dot(e.normal)+e.constant)/t;return n>=0?n:null}intersectPlane(e,t){const n=this.distanceToPlane(e);return n===null?null:this.at(n,t)}intersectsPlane(e){const t=e.distanceToPoint(this.origin);return t===0||e.normal.dot(this.direction)*t<0}intersectBox(e,t){let n,r,s,a,o,c;const l=1/this.direction.x,u=1/this.direction.y,d=1/this.direction.z,h=this.origin;return l>=0?(n=(e.min.x-h.x)*l,r=(e.max.x-h.x)*l):(n=(e.max.x-h.x)*l,r=(e.min.x-h.x)*l),u>=0?(s=(e.min.y-h.y)*u,a=(e.max.y-h.y)*u):(s=(e.max.y-h.y)*u,a=(e.min.y-h.y)*u),n>a||s>r||((s>n||isNaN(n))&&(n=s),(a<r||isNaN(r))&&(r=a),d>=0?(o=(e.min.z-h.z)*d,c=(e.max.z-h.z)*d):(o=(e.max.z-h.z)*d,c=(e.min.z-h.z)*d),n>c||o>r)||((o>n||n!==n)&&(n=o),(c<r||r!==r)&&(r=c),r<0)?null:this.at(n>=0?n:r,t)}intersectsBox(e){return this.intersectBox(e,Fn)!==null}intersectTriangle(e,t,n,r,s){Vs.subVectors(t,e),Or.subVectors(n,e),Ws.crossVectors(Vs,Or);let a=this.direction.dot(Ws),o;if(a>0){if(r)return null;o=1}else if(a<0)o=-1,a=-a;else return null;$n.subVectors(this.origin,e);const c=o*this.direction.dot(Or.crossVectors($n,Or));if(c<0)return null;const l=o*this.direction.dot(Vs.cross($n));if(l<0||c+l>a)return null;const u=-o*$n.dot(Ws);return u<0?null:this.at(u/a,s)}applyMatrix4(e){return this.origin.applyMatrix4(e),this.direction.transformDirection(e),this}equals(e){return e.origin.equals(this.origin)&&e.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}}class oc extends qi{constructor(e){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new rt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Rn,this.combine=ro,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.fog=e.fog,this}}const Xo=new _t,ci=new _s,Br=new Yi,Yo=new q,kr=new q,zr=new q,Gr=new q,Xs=new q,Hr=new q,qo=new q,Vr=new q;class _n extends Ot{constructor(e=new Qt,t=new oc){super(),this.isMesh=!0,this.type="Mesh",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),e.morphTargetInfluences!==void 0&&(this.morphTargetInfluences=e.morphTargetInfluences.slice()),e.morphTargetDictionary!==void 0&&(this.morphTargetDictionary=Object.assign({},e.morphTargetDictionary)),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const r=t[n[0]];if(r!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let s=0,a=r.length;s<a;s++){const o=r[s].name||String(s);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=s}}}}getVertexPosition(e,t){const n=this.geometry,r=n.attributes.position,s=n.morphAttributes.position,a=n.morphTargetsRelative;t.fromBufferAttribute(r,e);const o=this.morphTargetInfluences;if(s&&o){Hr.set(0,0,0);for(let c=0,l=s.length;c<l;c++){const u=o[c],d=s[c];u!==0&&(Xs.fromBufferAttribute(d,e),a?Hr.addScaledVector(Xs,u):Hr.addScaledVector(Xs.sub(t),u))}t.add(Hr)}return t}raycast(e,t){const n=this.geometry,r=this.material,s=this.matrixWorld;r!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),Br.copy(n.boundingSphere),Br.applyMatrix4(s),ci.copy(e.ray).recast(e.near),!(Br.containsPoint(ci.origin)===!1&&(ci.intersectSphere(Br,Yo)===null||ci.origin.distanceToSquared(Yo)>(e.far-e.near)**2))&&(Xo.copy(s).invert(),ci.copy(e.ray).applyMatrix4(Xo),!(n.boundingBox!==null&&ci.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(e,t,ci)))}_computeIntersections(e,t,n){let r;const s=this.geometry,a=this.material,o=s.index,c=s.attributes.position,l=s.attributes.uv,u=s.attributes.uv1,d=s.attributes.normal,h=s.groups,f=s.drawRange;if(o!==null)if(Array.isArray(a))for(let _=0,y=h.length;_<y;_++){const g=h[_],m=a[g.materialIndex],b=Math.max(g.start,f.start),w=Math.min(o.count,Math.min(g.start+g.count,f.start+f.count));for(let A=b,U=w;A<U;A+=3){const L=o.getX(A),N=o.getX(A+1),S=o.getX(A+2);r=Wr(this,m,e,n,l,u,d,L,N,S),r&&(r.faceIndex=Math.floor(A/3),r.face.materialIndex=g.materialIndex,t.push(r))}}else{const _=Math.max(0,f.start),y=Math.min(o.count,f.start+f.count);for(let g=_,m=y;g<m;g+=3){const b=o.getX(g),w=o.getX(g+1),A=o.getX(g+2);r=Wr(this,a,e,n,l,u,d,b,w,A),r&&(r.faceIndex=Math.floor(g/3),t.push(r))}}else if(c!==void 0)if(Array.isArray(a))for(let _=0,y=h.length;_<y;_++){const g=h[_],m=a[g.materialIndex],b=Math.max(g.start,f.start),w=Math.min(c.count,Math.min(g.start+g.count,f.start+f.count));for(let A=b,U=w;A<U;A+=3){const L=A,N=A+1,S=A+2;r=Wr(this,m,e,n,l,u,d,L,N,S),r&&(r.faceIndex=Math.floor(A/3),r.face.materialIndex=g.materialIndex,t.push(r))}}else{const _=Math.max(0,f.start),y=Math.min(c.count,f.start+f.count);for(let g=_,m=y;g<m;g+=3){const b=g,w=g+1,A=g+2;r=Wr(this,a,e,n,l,u,d,b,w,A),r&&(r.faceIndex=Math.floor(g/3),t.push(r))}}}}function Fh(i,e,t,n,r,s,a,o){let c;if(e.side===qt?c=n.intersectTriangle(a,s,r,!0,o):c=n.intersectTriangle(r,s,a,e.side===ri,o),c===null)return null;Vr.copy(o),Vr.applyMatrix4(i.matrixWorld);const l=t.ray.origin.distanceTo(Vr);return l<t.near||l>t.far?null:{distance:l,point:Vr.clone(),object:i}}function Wr(i,e,t,n,r,s,a,o,c,l){i.getVertexPosition(o,kr),i.getVertexPosition(c,zr),i.getVertexPosition(l,Gr);const u=Fh(i,e,t,n,kr,zr,Gr,qo);if(u){const d=new q;sn.getBarycoord(qo,kr,zr,Gr,d),r&&(u.uv=sn.getInterpolatedAttribute(r,o,c,l,d,new $e)),s&&(u.uv1=sn.getInterpolatedAttribute(s,o,c,l,d,new $e)),a&&(u.normal=sn.getInterpolatedAttribute(a,o,c,l,d,new q),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));const h={a:o,b:c,c:l,normal:new q,materialIndex:0};sn.getNormal(kr,zr,Gr,h.normal),u.face=h,u.barycoord=d}return u}class go extends Ht{constructor(e=null,t=1,n=1,r,s,a,o,c,l=Nt,u=Nt,d,h){super(null,a,o,c,l,u,r,s,d,h),this.isDataTexture=!0,this.image={data:e,width:t,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}}class Zo extends on{constructor(e,t,n,r=1){super(e,t,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=r}copy(e){return super.copy(e),this.meshPerAttribute=e.meshPerAttribute,this}toJSON(){const e=super.toJSON();return e.meshPerAttribute=this.meshPerAttribute,e.isInstancedBufferAttribute=!0,e}}const Li=new _t,$o=new _t,Xr=[],jo=new vi,Nh=new _t,ir=new _n,rr=new Yi;class Oh extends _n{constructor(e,t,n){super(e,t),this.isInstancedMesh=!0,this.instanceMatrix=new Zo(new Float32Array(n*16),16),this.previousInstanceMatrix=null,this.instanceColor=null,this.morphTexture=null,this.count=n,this.boundingBox=null,this.boundingSphere=null;for(let r=0;r<n;r++)this.setMatrixAt(r,Nh)}computeBoundingBox(){const e=this.geometry,t=this.count;this.boundingBox===null&&(this.boundingBox=new vi),e.boundingBox===null&&e.computeBoundingBox(),this.boundingBox.makeEmpty();for(let n=0;n<t;n++)this.getMatrixAt(n,Li),jo.copy(e.boundingBox).applyMatrix4(Li),this.boundingBox.union(jo)}computeBoundingSphere(){const e=this.geometry,t=this.count;this.boundingSphere===null&&(this.boundingSphere=new Yi),e.boundingSphere===null&&e.computeBoundingSphere(),this.boundingSphere.makeEmpty();for(let n=0;n<t;n++)this.getMatrixAt(n,Li),rr.copy(e.boundingSphere).applyMatrix4(Li),this.boundingSphere.union(rr)}copy(e,t){return super.copy(e,t),this.instanceMatrix.copy(e.instanceMatrix),e.previousInstanceMatrix!==null&&(this.previousInstanceMatrix=e.previousInstanceMatrix.clone()),e.morphTexture!==null&&(this.morphTexture=e.morphTexture.clone()),e.instanceColor!==null&&(this.instanceColor=e.instanceColor.clone()),this.count=e.count,e.boundingBox!==null&&(this.boundingBox=e.boundingBox.clone()),e.boundingSphere!==null&&(this.boundingSphere=e.boundingSphere.clone()),this}getColorAt(e,t){t.fromArray(this.instanceColor.array,e*3)}getMatrixAt(e,t){t.fromArray(this.instanceMatrix.array,e*16)}getMorphAt(e,t){const n=t.morphTargetInfluences,r=this.morphTexture.source.data.data,s=n.length+1,a=e*s+1;for(let o=0;o<n.length;o++)n[o]=r[a+o]}raycast(e,t){const n=this.matrixWorld,r=this.count;if(ir.geometry=this.geometry,ir.material=this.material,ir.material!==void 0&&(this.boundingSphere===null&&this.computeBoundingSphere(),rr.copy(this.boundingSphere),rr.applyMatrix4(n),e.ray.intersectsSphere(rr)!==!1))for(let s=0;s<r;s++){this.getMatrixAt(s,Li),$o.multiplyMatrices(n,Li),ir.matrixWorld=$o,ir.raycast(e,Xr);for(let a=0,o=Xr.length;a<o;a++){const c=Xr[a];c.instanceId=s,c.object=this,t.push(c)}Xr.length=0}}setColorAt(e,t){this.instanceColor===null&&(this.instanceColor=new Zo(new Float32Array(this.instanceMatrix.count*3).fill(1),3)),t.toArray(this.instanceColor.array,e*3)}setMatrixAt(e,t){t.toArray(this.instanceMatrix.array,e*16)}setMorphAt(e,t){const n=t.morphTargetInfluences,r=n.length+1;this.morphTexture===null&&(this.morphTexture=new go(new Float32Array(r*this.count),r,this.count,gi,Yt));const s=this.morphTexture.source.data.data;let a=0;for(let l=0;l<n.length;l++)a+=n[l];const o=this.geometry.morphTargetsRelative?1:1-a,c=r*e;s[c]=o,s.set(n,c+1)}updateMorphTargets(){}dispose(){this.dispatchEvent({type:"dispose"}),this.morphTexture!==null&&(this.morphTexture.dispose(),this.morphTexture=null)}}const Ys=new q,Bh=new q,kh=new Je;class Qn{constructor(e=new q(1,0,0),t=0){this.isPlane=!0,this.normal=e,this.constant=t}set(e,t){return this.normal.copy(e),this.constant=t,this}setComponents(e,t,n,r){return this.normal.set(e,t,n),this.constant=r,this}setFromNormalAndCoplanarPoint(e,t){return this.normal.copy(e),this.constant=-t.dot(this.normal),this}setFromCoplanarPoints(e,t,n){const r=Ys.subVectors(n,t).cross(Bh.subVectors(e,t)).normalize();return this.setFromNormalAndCoplanarPoint(r,e),this}copy(e){return this.normal.copy(e.normal),this.constant=e.constant,this}normalize(){const e=1/this.normal.length();return this.normal.multiplyScalar(e),this.constant*=e,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(e){return this.normal.dot(e)+this.constant}distanceToSphere(e){return this.distanceToPoint(e.center)-e.radius}projectPoint(e,t){return t.copy(e).addScaledVector(this.normal,-this.distanceToPoint(e))}intersectLine(e,t){const n=e.delta(Ys),r=this.normal.dot(n);if(r===0)return this.distanceToPoint(e.start)===0?t.copy(e.start):null;const s=-(e.start.dot(this.normal)+this.constant)/r;return s<0||s>1?null:t.copy(e.start).addScaledVector(n,s)}intersectsLine(e){const t=this.distanceToPoint(e.start),n=this.distanceToPoint(e.end);return t<0&&n>0||n<0&&t>0}intersectsBox(e){return e.intersectsPlane(this)}intersectsSphere(e){return e.intersectsPlane(this)}coplanarPoint(e){return e.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(e,t){const n=t||kh.getNormalMatrix(e),r=this.coplanarPoint(Ys).applyMatrix4(e),s=this.normal.applyMatrix3(n).normalize();return this.constant=-r.dot(s),this}translate(e){return this.constant-=e.dot(this.normal),this}equals(e){return e.normal.equals(this.normal)&&e.constant===this.constant}clone(){return new this.constructor().copy(this)}}const hi=new Yi,zh=new $e(.5,.5),Yr=new q;class _o{constructor(e=new Qn,t=new Qn,n=new Qn,r=new Qn,s=new Qn,a=new Qn){this.planes=[e,t,n,r,s,a]}set(e,t,n,r,s,a){const o=this.planes;return o[0].copy(e),o[1].copy(t),o[2].copy(n),o[3].copy(r),o[4].copy(s),o[5].copy(a),this}copy(e){const t=this.planes;for(let n=0;n<6;n++)t[n].copy(e.planes[n]);return this}setFromProjectionMatrix(e,t=Tn,n=!1){const r=this.planes,s=e.elements,a=s[0],o=s[1],c=s[2],l=s[3],u=s[4],d=s[5],h=s[6],f=s[7],_=s[8],y=s[9],g=s[10],m=s[11],b=s[12],w=s[13],A=s[14],U=s[15];if(r[0].setComponents(l-a,f-u,m-_,U-b).normalize(),r[1].setComponents(l+a,f+u,m+_,U+b).normalize(),r[2].setComponents(l+o,f+d,m+y,U+w).normalize(),r[3].setComponents(l-o,f-d,m-y,U-w).normalize(),n)r[4].setComponents(c,h,g,A).normalize(),r[5].setComponents(l-c,f-h,m-g,U-A).normalize();else if(r[4].setComponents(l-c,f-h,m-g,U-A).normalize(),t===Tn)r[5].setComponents(l+c,f+h,m+g,U+A).normalize();else if(t===_r)r[5].setComponents(c,h,g,A).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+t);return this}intersectsObject(e){if(e.boundingSphere!==void 0)e.boundingSphere===null&&e.computeBoundingSphere(),hi.copy(e.boundingSphere).applyMatrix4(e.matrixWorld);else{const t=e.geometry;t.boundingSphere===null&&t.computeBoundingSphere(),hi.copy(t.boundingSphere).applyMatrix4(e.matrixWorld)}return this.intersectsSphere(hi)}intersectsSprite(e){hi.center.set(0,0,0);const t=zh.distanceTo(e.center);return hi.radius=.7071067811865476+t,hi.applyMatrix4(e.matrixWorld),this.intersectsSphere(hi)}intersectsSphere(e){const t=this.planes,n=e.center,r=-e.radius;for(let s=0;s<6;s++)if(t[s].distanceToPoint(n)<r)return!1;return!0}intersectsBox(e){const t=this.planes;for(let n=0;n<6;n++){const r=t[n];if(Yr.x=r.normal.x>0?e.max.x:e.min.x,Yr.y=r.normal.y>0?e.max.y:e.min.y,Yr.z=r.normal.z>0?e.max.z:e.min.z,r.distanceToPoint(Yr)<0)return!1}return!0}containsPoint(e){const t=this.planes;for(let n=0;n<6;n++)if(t[n].distanceToPoint(e)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}}class lc extends qi{constructor(e){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new rt(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.map=e.map,this.linewidth=e.linewidth,this.linecap=e.linecap,this.linejoin=e.linejoin,this.fog=e.fog,this}}const ps=new q,ms=new q,Ko=new _t,sr=new _s,qr=new Yi,qs=new q,Jo=new q;class Gh extends Ot{constructor(e=new Qt,t=new lc){super(),this.isLine=!0,this.type="Line",this.geometry=e,this.material=t,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(e,t){return super.copy(e,t),this.material=Array.isArray(e.material)?e.material.slice():e.material,this.geometry=e.geometry,this}computeLineDistances(){const e=this.geometry;if(e.index===null){const t=e.attributes.position,n=[0];for(let r=1,s=t.count;r<s;r++)ps.fromBufferAttribute(t,r-1),ms.fromBufferAttribute(t,r),n[r]=n[r-1],n[r]+=ps.distanceTo(ms);e.setAttribute("lineDistance",new It(n,1))}else Xe("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}raycast(e,t){const n=this.geometry,r=this.matrixWorld,s=e.params.Line.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),qr.copy(n.boundingSphere),qr.applyMatrix4(r),qr.radius+=s,e.ray.intersectsSphere(qr)===!1)return;Ko.copy(r).invert(),sr.copy(e.ray).applyMatrix4(Ko);const o=s/((this.scale.x+this.scale.y+this.scale.z)/3),c=o*o,l=this.isLineSegments?2:1,u=n.index,h=n.attributes.position;if(u!==null){const f=Math.max(0,a.start),_=Math.min(u.count,a.start+a.count);for(let y=f,g=_-1;y<g;y+=l){const m=u.getX(y),b=u.getX(y+1),w=Zr(this,e,sr,c,m,b,y);w&&t.push(w)}if(this.isLineLoop){const y=u.getX(_-1),g=u.getX(f),m=Zr(this,e,sr,c,y,g,_-1);m&&t.push(m)}}else{const f=Math.max(0,a.start),_=Math.min(h.count,a.start+a.count);for(let y=f,g=_-1;y<g;y+=l){const m=Zr(this,e,sr,c,y,y+1,y);m&&t.push(m)}if(this.isLineLoop){const y=Zr(this,e,sr,c,_-1,f,_-1);y&&t.push(y)}}}updateMorphTargets(){const t=this.geometry.morphAttributes,n=Object.keys(t);if(n.length>0){const r=t[n[0]];if(r!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let s=0,a=r.length;s<a;s++){const o=r[s].name||String(s);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=s}}}}}function Zr(i,e,t,n,r,s,a){const o=i.geometry.attributes.position;if(ps.fromBufferAttribute(o,r),ms.fromBufferAttribute(o,s),t.distanceSqToSegment(ps,ms,qs,Jo)>n)return;qs.applyMatrix4(i.matrixWorld);const l=e.ray.origin.distanceTo(qs);if(!(l<e.near||l>e.far))return{distance:l,point:Jo.clone().applyMatrix4(i.matrixWorld),index:a,face:null,faceIndex:null,barycoord:null,object:i}}const Qo=new q,el=new q;class Hh extends Gh{constructor(e,t){super(e,t),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){const e=this.geometry;if(e.index===null){const t=e.attributes.position,n=[];for(let r=0,s=t.count;r<s;r+=2)Qo.fromBufferAttribute(t,r),el.fromBufferAttribute(t,r+1),n[r]=r===0?0:n[r-1],n[r+1]=n[r]+Qo.distanceTo(el);e.setAttribute("lineDistance",new It(n,1))}else Xe("LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}}class cc extends Ht{constructor(e=[],t=_i,n,r,s,a,o,c,l,u){super(e,t,n,r,s,a,o,c,l,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(e){this.image=e}}class xr extends Ht{constructor(e,t,n=Cn,r,s,a,o=Nt,c=Nt,l,u=Hn,d=1){if(u!==Hn&&u!==mi)throw new Error("DepthTexture format must be either THREE.DepthFormat or THREE.DepthStencilFormat");const h={width:e,height:t,depth:d};super(h,r,s,a,o,c,u,n,l),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(e){return super.copy(e),this.source=new po(Object.assign({},e.image)),this.compareFunction=e.compareFunction,this}toJSON(e){const t=super.toJSON(e);return this.compareFunction!==null&&(t.compareFunction=this.compareFunction),t}}class Vh extends xr{constructor(e,t=Cn,n=_i,r,s,a=Nt,o=Nt,c,l=Hn){const u={width:e,height:e,depth:1},d=[u,u,u,u,u,u];super(e,e,t,n,r,s,a,o,c,l),this.image=d,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(e){this.image=e}}class hc extends Ht{constructor(e=null){super(),this.sourceTexture=e,this.isExternalTexture=!0}copy(e){return super.copy(e),this.sourceTexture=e.sourceTexture,this}}class ii extends Qt{constructor(e=1,t=1,n=1,r=1,s=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:e,height:t,depth:n,widthSegments:r,heightSegments:s,depthSegments:a};const o=this;r=Math.floor(r),s=Math.floor(s),a=Math.floor(a);const c=[],l=[],u=[],d=[];let h=0,f=0;_("z","y","x",-1,-1,n,t,e,a,s,0),_("z","y","x",1,-1,n,t,-e,a,s,1),_("x","z","y",1,1,e,n,t,r,a,2),_("x","z","y",1,-1,e,n,-t,r,a,3),_("x","y","z",1,-1,e,t,n,r,s,4),_("x","y","z",-1,-1,e,t,-n,r,s,5),this.setIndex(c),this.setAttribute("position",new It(l,3)),this.setAttribute("normal",new It(u,3)),this.setAttribute("uv",new It(d,2));function _(y,g,m,b,w,A,U,L,N,S,T){const G=A/N,D=U/S,O=A/2,V=U/2,K=L/2,Y=N+1,Z=S+1;let X=0,fe=0;const oe=new q;for(let ye=0;ye<Z;ye++){const Ae=ye*D-V;for(let ve=0;ve<Y;ve++){const Ge=ve*G-O;oe[y]=Ge*b,oe[g]=Ae*w,oe[m]=K,l.push(oe.x,oe.y,oe.z),oe[y]=0,oe[g]=0,oe[m]=L>0?1:-1,u.push(oe.x,oe.y,oe.z),d.push(ve/N),d.push(1-ye/S),X+=1}}for(let ye=0;ye<S;ye++)for(let Ae=0;Ae<N;Ae++){const ve=h+Ae+Y*ye,Ge=h+Ae+Y*(ye+1),st=h+(Ae+1)+Y*(ye+1),_e=h+(Ae+1)+Y*ye;c.push(ve,Ge,_e),c.push(Ge,st,_e),fe+=6}o.addGroup(f,fe,T),f+=fe,h+=X}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new ii(e.width,e.height,e.depth,e.widthSegments,e.heightSegments,e.depthSegments)}}class xo extends Qt{constructor(e=1,t=1,n=1,r=32,s=1,a=!1,o=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:e,radiusBottom:t,height:n,radialSegments:r,heightSegments:s,openEnded:a,thetaStart:o,thetaLength:c};const l=this;r=Math.floor(r),s=Math.floor(s);const u=[],d=[],h=[],f=[];let _=0;const y=[],g=n/2;let m=0;b(),a===!1&&(e>0&&w(!0),t>0&&w(!1)),this.setIndex(u),this.setAttribute("position",new It(d,3)),this.setAttribute("normal",new It(h,3)),this.setAttribute("uv",new It(f,2));function b(){const A=new q,U=new q;let L=0;const N=(t-e)/n;for(let S=0;S<=s;S++){const T=[],G=S/s,D=G*(t-e)+e;for(let O=0;O<=r;O++){const V=O/r,K=V*c+o,Y=Math.sin(K),Z=Math.cos(K);U.x=D*Y,U.y=-G*n+g,U.z=D*Z,d.push(U.x,U.y,U.z),A.set(Y,N,Z).normalize(),h.push(A.x,A.y,A.z),f.push(V,1-G),T.push(_++)}y.push(T)}for(let S=0;S<r;S++)for(let T=0;T<s;T++){const G=y[T][S],D=y[T+1][S],O=y[T+1][S+1],V=y[T][S+1];(e>0||T!==0)&&(u.push(G,D,V),L+=3),(t>0||T!==s-1)&&(u.push(D,O,V),L+=3)}l.addGroup(m,L,0),m+=L}function w(A){const U=_,L=new $e,N=new q;let S=0;const T=A===!0?e:t,G=A===!0?1:-1;for(let O=1;O<=r;O++)d.push(0,g*G,0),h.push(0,G,0),f.push(.5,.5),_++;const D=_;for(let O=0;O<=r;O++){const K=O/r*c+o,Y=Math.cos(K),Z=Math.sin(K);N.x=T*Z,N.y=g*G,N.z=T*Y,d.push(N.x,N.y,N.z),h.push(0,G,0),L.x=Y*.5+.5,L.y=Z*.5*G+.5,f.push(L.x,L.y),_++}for(let O=0;O<r;O++){const V=U+O,K=D+O;A===!0?u.push(K,K+1,V):u.push(K+1,K,V),S+=3}l.addGroup(m,S,A===!0?1:2),m+=S}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new xo(e.radiusTop,e.radiusBottom,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}class vo extends xo{constructor(e=1,t=1,n=32,r=1,s=!1,a=0,o=Math.PI*2){super(0,e,t,n,r,s,a,o),this.type="ConeGeometry",this.parameters={radius:e,height:t,radialSegments:n,heightSegments:r,openEnded:s,thetaStart:a,thetaLength:o}}static fromJSON(e){return new vo(e.radius,e.height,e.radialSegments,e.heightSegments,e.openEnded,e.thetaStart,e.thetaLength)}}const $r=new q,jr=new q,Zs=new q,Kr=new sn;class Wh extends Qt{constructor(e=null,t=1){if(super(),this.type="EdgesGeometry",this.parameters={geometry:e,thresholdAngle:t},e!==null){const r=Math.pow(10,4),s=Math.cos(fr*t),a=e.getIndex(),o=e.getAttribute("position"),c=a?a.count:o.count,l=[0,0,0],u=["a","b","c"],d=new Array(3),h={},f=[];for(let _=0;_<c;_+=3){a?(l[0]=a.getX(_),l[1]=a.getX(_+1),l[2]=a.getX(_+2)):(l[0]=_,l[1]=_+1,l[2]=_+2);const{a:y,b:g,c:m}=Kr;if(y.fromBufferAttribute(o,l[0]),g.fromBufferAttribute(o,l[1]),m.fromBufferAttribute(o,l[2]),Kr.getNormal(Zs),d[0]=`${Math.round(y.x*r)},${Math.round(y.y*r)},${Math.round(y.z*r)}`,d[1]=`${Math.round(g.x*r)},${Math.round(g.y*r)},${Math.round(g.z*r)}`,d[2]=`${Math.round(m.x*r)},${Math.round(m.y*r)},${Math.round(m.z*r)}`,!(d[0]===d[1]||d[1]===d[2]||d[2]===d[0]))for(let b=0;b<3;b++){const w=(b+1)%3,A=d[b],U=d[w],L=Kr[u[b]],N=Kr[u[w]],S=`${A}_${U}`,T=`${U}_${A}`;T in h&&h[T]?(Zs.dot(h[T].normal)<=s&&(f.push(L.x,L.y,L.z),f.push(N.x,N.y,N.z)),h[T]=null):S in h||(h[S]={index0:l[b],index1:l[w],normal:Zs.clone()})}}for(const _ in h)if(h[_]){const{index0:y,index1:g}=h[_];$r.fromBufferAttribute(o,y),jr.fromBufferAttribute(o,g),f.push($r.x,$r.y,$r.z),f.push(jr.x,jr.y,jr.z)}this.setAttribute("position",new It(f,3))}}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}}class xs extends Qt{constructor(e=1,t=1,n=1,r=1){super(),this.type="PlaneGeometry",this.parameters={width:e,height:t,widthSegments:n,heightSegments:r};const s=e/2,a=t/2,o=Math.floor(n),c=Math.floor(r),l=o+1,u=c+1,d=e/o,h=t/c,f=[],_=[],y=[],g=[];for(let m=0;m<u;m++){const b=m*h-a;for(let w=0;w<l;w++){const A=w*d-s;_.push(A,-b,0),y.push(0,0,1),g.push(w/o),g.push(1-m/c)}}for(let m=0;m<c;m++)for(let b=0;b<o;b++){const w=b+l*m,A=b+l*(m+1),U=b+1+l*(m+1),L=b+1+l*m;f.push(w,A,L),f.push(A,U,L)}this.setIndex(f),this.setAttribute("position",new It(_,3)),this.setAttribute("normal",new It(y,3)),this.setAttribute("uv",new It(g,2))}copy(e){return super.copy(e),this.parameters=Object.assign({},e.parameters),this}static fromJSON(e){return new xs(e.width,e.height,e.widthSegments,e.heightSegments)}}function Xi(i){const e={};for(const t in i){e[t]={};for(const n in i[t]){const r=i[t][n];r&&(r.isColor||r.isMatrix3||r.isMatrix4||r.isVector2||r.isVector3||r.isVector4||r.isTexture||r.isQuaternion)?r.isRenderTargetTexture?(Xe("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),e[t][n]=null):e[t][n]=r.clone():Array.isArray(r)?e[t][n]=r.slice():e[t][n]=r}}return e}function zt(i){const e={};for(let t=0;t<i.length;t++){const n=Xi(i[t]);for(const r in n)e[r]=n[r]}return e}function Xh(i){const e=[];for(let t=0;t<i.length;t++)e.push(i[t].clone());return e}function uc(i){const e=i.getRenderTarget();return e===null?i.outputColorSpace:e.isXRRenderTarget===!0?e.texture.colorSpace:lt.workingColorSpace}const Yh={clone:Xi,merge:zt};var qh=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,Zh=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`;class Pn extends qi{constructor(e){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=qh,this.fragmentShader=Zh,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,e!==void 0&&this.setValues(e)}copy(e){return super.copy(e),this.fragmentShader=e.fragmentShader,this.vertexShader=e.vertexShader,this.uniforms=Xi(e.uniforms),this.uniformsGroups=Xh(e.uniformsGroups),this.defines=Object.assign({},e.defines),this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.fog=e.fog,this.lights=e.lights,this.clipping=e.clipping,this.extensions=Object.assign({},e.extensions),this.glslVersion=e.glslVersion,this.defaultAttributeValues=Object.assign({},e.defaultAttributeValues),this.index0AttributeName=e.index0AttributeName,this.uniformsNeedUpdate=e.uniformsNeedUpdate,this}toJSON(e){const t=super.toJSON(e);t.glslVersion=this.glslVersion,t.uniforms={};for(const r in this.uniforms){const a=this.uniforms[r].value;a&&a.isTexture?t.uniforms[r]={type:"t",value:a.toJSON(e).uuid}:a&&a.isColor?t.uniforms[r]={type:"c",value:a.getHex()}:a&&a.isVector2?t.uniforms[r]={type:"v2",value:a.toArray()}:a&&a.isVector3?t.uniforms[r]={type:"v3",value:a.toArray()}:a&&a.isVector4?t.uniforms[r]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?t.uniforms[r]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?t.uniforms[r]={type:"m4",value:a.toArray()}:t.uniforms[r]={value:a}}Object.keys(this.defines).length>0&&(t.defines=this.defines),t.vertexShader=this.vertexShader,t.fragmentShader=this.fragmentShader,t.lights=this.lights,t.clipping=this.clipping;const n={};for(const r in this.extensions)this.extensions[r]===!0&&(n[r]=!0);return Object.keys(n).length>0&&(t.extensions=n),t}}class $h extends Pn{constructor(e){super(e),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}}class jh extends qi{constructor(e){super(),this.isMeshPhongMaterial=!0,this.type="MeshPhongMaterial",this.color=new rt(16777215),this.specular=new rt(1118481),this.shininess=30,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new rt(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=tc,this.normalScale=new $e(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new Rn,this.combine=ro,this.reflectivity=1,this.envMapIntensity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(e)}copy(e){return super.copy(e),this.color.copy(e.color),this.specular.copy(e.specular),this.shininess=e.shininess,this.map=e.map,this.lightMap=e.lightMap,this.lightMapIntensity=e.lightMapIntensity,this.aoMap=e.aoMap,this.aoMapIntensity=e.aoMapIntensity,this.emissive.copy(e.emissive),this.emissiveMap=e.emissiveMap,this.emissiveIntensity=e.emissiveIntensity,this.bumpMap=e.bumpMap,this.bumpScale=e.bumpScale,this.normalMap=e.normalMap,this.normalMapType=e.normalMapType,this.normalScale.copy(e.normalScale),this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.specularMap=e.specularMap,this.alphaMap=e.alphaMap,this.envMap=e.envMap,this.envMapRotation.copy(e.envMapRotation),this.combine=e.combine,this.reflectivity=e.reflectivity,this.envMapIntensity=e.envMapIntensity,this.refractionRatio=e.refractionRatio,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this.wireframeLinecap=e.wireframeLinecap,this.wireframeLinejoin=e.wireframeLinejoin,this.flatShading=e.flatShading,this.fog=e.fog,this}}class Kh extends qi{constructor(e){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=Qc,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(e)}copy(e){return super.copy(e),this.depthPacking=e.depthPacking,this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this.wireframe=e.wireframe,this.wireframeLinewidth=e.wireframeLinewidth,this}}class Jh extends qi{constructor(e){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(e)}copy(e){return super.copy(e),this.map=e.map,this.alphaMap=e.alphaMap,this.displacementMap=e.displacementMap,this.displacementScale=e.displacementScale,this.displacementBias=e.displacementBias,this}}const tl={enabled:!1,files:{},add:function(i,e){this.enabled!==!1&&(nl(i)||(this.files[i]=e))},get:function(i){if(this.enabled!==!1&&!nl(i))return this.files[i]},remove:function(i){delete this.files[i]},clear:function(){this.files={}}};function nl(i){try{const e=i.slice(i.indexOf(":")+1);return new URL(e).protocol==="blob:"}catch{return!1}}class Qh{constructor(e,t,n){const r=this;let s=!1,a=0,o=0,c;const l=[];this.onStart=void 0,this.onLoad=e,this.onProgress=t,this.onError=n,this._abortController=null,this.itemStart=function(u){o++,s===!1&&r.onStart!==void 0&&r.onStart(u,a,o),s=!0},this.itemEnd=function(u){a++,r.onProgress!==void 0&&r.onProgress(u,a,o),a===o&&(s=!1,r.onLoad!==void 0&&r.onLoad())},this.itemError=function(u){r.onError!==void 0&&r.onError(u)},this.resolveURL=function(u){return c?c(u):u},this.setURLModifier=function(u){return c=u,this},this.addHandler=function(u,d){return l.push(u,d),this},this.removeHandler=function(u){const d=l.indexOf(u);return d!==-1&&l.splice(d,2),this},this.getHandler=function(u){for(let d=0,h=l.length;d<h;d+=2){const f=l[d],_=l[d+1];if(f.global&&(f.lastIndex=0),f.test(u))return _}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}}const eu=new Qh;class So{constructor(e){this.manager=e!==void 0?e:eu,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(e,t){const n=this;return new Promise(function(r,s){n.load(e,r,t,s)})}parse(){}setCrossOrigin(e){return this.crossOrigin=e,this}setWithCredentials(e){return this.withCredentials=e,this}setPath(e){return this.path=e,this}setResourcePath(e){return this.resourcePath=e,this}setRequestHeader(e){return this.requestHeader=e,this}abort(){return this}}So.DEFAULT_MATERIAL_NAME="__DEFAULT";const Nn={};class tu extends Error{constructor(e,t){super(e),this.response=t}}class nu extends So{constructor(e){super(e),this.mimeType="",this.responseType="",this._abortController=new AbortController}load(e,t,n,r){e===void 0&&(e=""),this.path!==void 0&&(e=this.path+e),e=this.manager.resolveURL(e);const s=tl.get(`file:${e}`);if(s!==void 0)return this.manager.itemStart(e),setTimeout(()=>{t&&t(s),this.manager.itemEnd(e)},0),s;if(Nn[e]!==void 0){Nn[e].push({onLoad:t,onProgress:n,onError:r});return}Nn[e]=[],Nn[e].push({onLoad:t,onProgress:n,onError:r});const a=new Request(e,{headers:new Headers(this.requestHeader),credentials:this.withCredentials?"include":"same-origin",signal:typeof AbortSignal.any=="function"?AbortSignal.any([this._abortController.signal,this.manager.abortController.signal]):this._abortController.signal}),o=this.mimeType,c=this.responseType;fetch(a).then(l=>{if(l.status===200||l.status===0){if(l.status===0&&Xe("FileLoader: HTTP Status 0 received."),typeof ReadableStream>"u"||l.body===void 0||l.body.getReader===void 0)return l;const u=Nn[e],d=l.body.getReader(),h=l.headers.get("X-File-Size")||l.headers.get("Content-Length"),f=h?parseInt(h):0,_=f!==0;let y=0;const g=new ReadableStream({start(m){b();function b(){d.read().then(({done:w,value:A})=>{if(w)m.close();else{y+=A.byteLength;const U=new ProgressEvent("progress",{lengthComputable:_,loaded:y,total:f});for(let L=0,N=u.length;L<N;L++){const S=u[L];S.onProgress&&S.onProgress(U)}m.enqueue(A),b()}},w=>{m.error(w)})}}});return new Response(g)}else throw new tu(`fetch for "${l.url}" responded with ${l.status}: ${l.statusText}`,l)}).then(l=>{switch(c){case"arraybuffer":return l.arrayBuffer();case"blob":return l.blob();case"document":return l.text().then(u=>new DOMParser().parseFromString(u,o));case"json":return l.json();default:if(o==="")return l.text();{const d=/charset="?([^;"\s]*)"?/i.exec(o),h=d&&d[1]?d[1].toLowerCase():void 0,f=new TextDecoder(h);return l.arrayBuffer().then(_=>f.decode(_))}}}).then(l=>{tl.add(`file:${e}`,l);const u=Nn[e];delete Nn[e];for(let d=0,h=u.length;d<h;d++){const f=u[d];f.onLoad&&f.onLoad(l)}}).catch(l=>{const u=Nn[e];if(u===void 0)throw this.manager.itemError(e),l;delete Nn[e];for(let d=0,h=u.length;d<h;d++){const f=u[d];f.onError&&f.onError(l)}this.manager.itemError(e)}).finally(()=>{this.manager.itemEnd(e)}),this.manager.itemStart(e)}setResponseType(e){return this.responseType=e,this}setMimeType(e){return this.mimeType=e,this}abort(){return this._abortController.abort(),this._abortController=new AbortController,this}}class iu extends So{constructor(e){super(e)}load(e,t,n,r){const s=this,a=new go,o=new nu(this.manager);return o.setResponseType("arraybuffer"),o.setRequestHeader(this.requestHeader),o.setPath(this.path),o.setWithCredentials(s.withCredentials),o.load(e,function(c){let l;try{l=s.parse(c)}catch(u){if(r!==void 0)r(u);else{u(u);return}}l.image!==void 0?a.image=l.image:l.data!==void 0&&(a.image.width=l.width,a.image.height=l.height,a.image.data=l.data),a.wrapS=l.wrapS!==void 0?l.wrapS:gn,a.wrapT=l.wrapT!==void 0?l.wrapT:gn,a.magFilter=l.magFilter!==void 0?l.magFilter:bt,a.minFilter=l.minFilter!==void 0?l.minFilter:bt,a.anisotropy=l.anisotropy!==void 0?l.anisotropy:1,l.colorSpace!==void 0&&(a.colorSpace=l.colorSpace),l.flipY!==void 0&&(a.flipY=l.flipY),l.format!==void 0&&(a.format=l.format),l.type!==void 0&&(a.type=l.type),l.mipmaps!==void 0&&(a.mipmaps=l.mipmaps,a.minFilter=ti),l.mipmapCount===1&&(a.minFilter=bt),l.generateMipmaps!==void 0&&(a.generateMipmaps=l.generateMipmaps),a.needsUpdate=!0,t&&t(a,l)},n,r),a}}class dc extends Ot{constructor(e,t=1){super(),this.isLight=!0,this.type="Light",this.color=new rt(e),this.intensity=t}dispose(){this.dispatchEvent({type:"dispose"})}copy(e,t){return super.copy(e,t),this.color.copy(e.color),this.intensity=e.intensity,this}toJSON(e){const t=super.toJSON(e);return t.object.color=this.color.getHex(),t.object.intensity=this.intensity,t}}const $s=new _t,il=new q,rl=new q;class ru{constructor(e){this.camera=e,this.intensity=1,this.bias=0,this.biasNode=null,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new $e(512,512),this.mapType=Kt,this.map=null,this.mapPass=null,this.matrix=new _t,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new _o,this._frameExtents=new $e(1,1),this._viewportCount=1,this._viewports=[new Et(0,0,1,1)]}getViewportCount(){return this._viewportCount}getFrustum(){return this._frustum}updateMatrices(e){const t=this.camera,n=this.matrix;il.setFromMatrixPosition(e.matrixWorld),t.position.copy(il),rl.setFromMatrixPosition(e.target.matrixWorld),t.lookAt(rl),t.updateMatrixWorld(),$s.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),this._frustum.setFromProjectionMatrix($s,t.coordinateSystem,t.reversedDepth),t.coordinateSystem===_r||t.reversedDepth?n.set(.5,0,0,.5,0,.5,0,.5,0,0,1,0,0,0,0,1):n.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),n.multiply($s)}getViewport(e){return this._viewports[e]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(e){return this.camera=e.camera.clone(),this.intensity=e.intensity,this.bias=e.bias,this.radius=e.radius,this.autoUpdate=e.autoUpdate,this.needsUpdate=e.needsUpdate,this.normalBias=e.normalBias,this.blurSamples=e.blurSamples,this.mapSize.copy(e.mapSize),this.biasNode=e.biasNode,this}clone(){return new this.constructor().copy(this)}toJSON(){const e={};return this.intensity!==1&&(e.intensity=this.intensity),this.bias!==0&&(e.bias=this.bias),this.normalBias!==0&&(e.normalBias=this.normalBias),this.radius!==1&&(e.radius=this.radius),(this.mapSize.x!==512||this.mapSize.y!==512)&&(e.mapSize=this.mapSize.toArray()),e.camera=this.camera.toJSON(!1).object,delete e.camera.matrix,e}}const Jr=new q,Qr=new si,Mn=new q;class fc extends Ot{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new _t,this.projectionMatrix=new _t,this.projectionMatrixInverse=new _t,this.coordinateSystem=Tn,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(e,t){return super.copy(e,t),this.matrixWorldInverse.copy(e.matrixWorldInverse),this.projectionMatrix.copy(e.projectionMatrix),this.projectionMatrixInverse.copy(e.projectionMatrixInverse),this.coordinateSystem=e.coordinateSystem,this}getWorldDirection(e){return super.getWorldDirection(e).negate()}updateMatrixWorld(e){super.updateMatrixWorld(e),this.matrixWorld.decompose(Jr,Qr,Mn),Mn.x===1&&Mn.y===1&&Mn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Jr,Qr,Mn.set(1,1,1)).invert()}updateWorldMatrix(e,t){super.updateWorldMatrix(e,t),this.matrixWorld.decompose(Jr,Qr,Mn),Mn.x===1&&Mn.y===1&&Mn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Jr,Qr,Mn.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}}const jn=new q,sl=new $e,al=new $e;class rn extends fc{constructor(e=50,t=1,n=.1,r=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=e,this.zoom=1,this.near=n,this.far=r,this.focus=10,this.aspect=t,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.fov=e.fov,this.zoom=e.zoom,this.near=e.near,this.far=e.far,this.focus=e.focus,this.aspect=e.aspect,this.view=e.view===null?null:Object.assign({},e.view),this.filmGauge=e.filmGauge,this.filmOffset=e.filmOffset,this}setFocalLength(e){const t=.5*this.getFilmHeight()/e;this.fov=ja*2*Math.atan(t),this.updateProjectionMatrix()}getFocalLength(){const e=Math.tan(fr*.5*this.fov);return .5*this.getFilmHeight()/e}getEffectiveFOV(){return ja*2*Math.atan(Math.tan(fr*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(e,t,n){jn.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),t.set(jn.x,jn.y).multiplyScalar(-e/jn.z),jn.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(jn.x,jn.y).multiplyScalar(-e/jn.z)}getViewSize(e,t){return this.getViewBounds(e,sl,al),t.subVectors(al,sl)}setViewOffset(e,t,n,r,s,a){this.aspect=e/t,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=r,this.view.width=s,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=this.near;let t=e*Math.tan(fr*.5*this.fov)/this.zoom,n=2*t,r=this.aspect*n,s=-.5*r;const a=this.view;if(this.view!==null&&this.view.enabled){const c=a.fullWidth,l=a.fullHeight;s+=a.offsetX*r/c,t-=a.offsetY*n/l,r*=a.width/c,n*=a.height/l}const o=this.filmOffset;o!==0&&(s+=e*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(s,s+r,t,t-n,e,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.fov=this.fov,t.object.zoom=this.zoom,t.object.near=this.near,t.object.far=this.far,t.object.focus=this.focus,t.object.aspect=this.aspect,this.view!==null&&(t.object.view=Object.assign({},this.view)),t.object.filmGauge=this.filmGauge,t.object.filmOffset=this.filmOffset,t}}class Mo extends fc{constructor(e=-1,t=1,n=1,r=-1,s=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=e,this.right=t,this.top=n,this.bottom=r,this.near=s,this.far=a,this.updateProjectionMatrix()}copy(e,t){return super.copy(e,t),this.left=e.left,this.right=e.right,this.top=e.top,this.bottom=e.bottom,this.near=e.near,this.far=e.far,this.zoom=e.zoom,this.view=e.view===null?null:Object.assign({},e.view),this}setViewOffset(e,t,n,r,s,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=e,this.view.fullHeight=t,this.view.offsetX=n,this.view.offsetY=r,this.view.width=s,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){const e=(this.right-this.left)/(2*this.zoom),t=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,r=(this.top+this.bottom)/2;let s=n-e,a=n+e,o=r+t,c=r-t;if(this.view!==null&&this.view.enabled){const l=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;s+=l*this.view.offsetX,a=s+l*this.view.width,o-=u*this.view.offsetY,c=o-u*this.view.height}this.projectionMatrix.makeOrthographic(s,a,o,c,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(e){const t=super.toJSON(e);return t.object.zoom=this.zoom,t.object.left=this.left,t.object.right=this.right,t.object.top=this.top,t.object.bottom=this.bottom,t.object.near=this.near,t.object.far=this.far,this.view!==null&&(t.object.view=Object.assign({},this.view)),t}}class su extends ru{constructor(){super(new Mo(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}}class au extends dc{constructor(e,t){super(e,t),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(Ot.DEFAULT_UP),this.updateMatrix(),this.target=new Ot,this.shadow=new su}dispose(){super.dispose(),this.shadow.dispose()}copy(e){return super.copy(e),this.target=e.target.clone(),this.shadow=e.shadow.clone(),this}toJSON(e){const t=super.toJSON(e);return t.object.shadow=this.shadow.toJSON(),t.object.target=this.target.uuid,t}}class ou extends dc{constructor(e,t){super(e,t),this.isAmbientLight=!0,this.type="AmbientLight"}}const Ui=-90,Fi=1;class lu extends Ot{constructor(e,t,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;const r=new rn(Ui,Fi,e,t);r.layers=this.layers,this.add(r);const s=new rn(Ui,Fi,e,t);s.layers=this.layers,this.add(s);const a=new rn(Ui,Fi,e,t);a.layers=this.layers,this.add(a);const o=new rn(Ui,Fi,e,t);o.layers=this.layers,this.add(o);const c=new rn(Ui,Fi,e,t);c.layers=this.layers,this.add(c);const l=new rn(Ui,Fi,e,t);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){const e=this.coordinateSystem,t=this.children.concat(),[n,r,s,a,o,c]=t;for(const l of t)this.remove(l);if(e===Tn)n.up.set(0,1,0),n.lookAt(1,0,0),r.up.set(0,1,0),r.lookAt(-1,0,0),s.up.set(0,0,-1),s.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(e===_r)n.up.set(0,-1,0),n.lookAt(-1,0,0),r.up.set(0,-1,0),r.lookAt(1,0,0),s.up.set(0,0,1),s.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+e);for(const l of t)this.add(l),l.updateMatrixWorld()}update(e,t){this.parent===null&&this.updateMatrixWorld();const{renderTarget:n,activeMipmapLevel:r}=this;this.coordinateSystem!==e.coordinateSystem&&(this.coordinateSystem=e.coordinateSystem,this.updateCoordinateSystem());const[s,a,o,c,l,u]=this.children,d=e.getRenderTarget(),h=e.getActiveCubeFace(),f=e.getActiveMipmapLevel(),_=e.xr.enabled;e.xr.enabled=!1;const y=n.texture.generateMipmaps;n.texture.generateMipmaps=!1;let g=!1;e.isWebGLRenderer===!0?g=e.state.buffers.depth.getReversed():g=e.reversedDepthBuffer,e.setRenderTarget(n,0,r),g&&e.autoClear===!1&&e.clearDepth(),e.render(t,s),e.setRenderTarget(n,1,r),g&&e.autoClear===!1&&e.clearDepth(),e.render(t,a),e.setRenderTarget(n,2,r),g&&e.autoClear===!1&&e.clearDepth(),e.render(t,o),e.setRenderTarget(n,3,r),g&&e.autoClear===!1&&e.clearDepth(),e.render(t,c),e.setRenderTarget(n,4,r),g&&e.autoClear===!1&&e.clearDepth(),e.render(t,l),n.texture.generateMipmaps=y,e.setRenderTarget(n,5,r),g&&e.autoClear===!1&&e.clearDepth(),e.render(t,u),e.setRenderTarget(d,h,f),e.xr.enabled=_,n.texture.needsPMREMUpdate=!0}}class cu extends rn{constructor(e=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=e}}const ol=new _t;class hu{constructor(e,t,n=0,r=1/0){this.ray=new _s(e,t),this.near=n,this.far=r,this.camera=null,this.layers=new mo,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(e,t){this.ray.set(e,t)}setFromCamera(e,t){t.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(e.x,e.y,.5).unproject(t).sub(this.ray.origin).normalize(),this.camera=t):t.isOrthographicCamera?(this.ray.origin.set(e.x,e.y,(t.near+t.far)/(t.near-t.far)).unproject(t),this.ray.direction.set(0,0,-1).transformDirection(t.matrixWorld),this.camera=t):ot("Raycaster: Unsupported camera type: "+t.type)}setFromXRController(e){return ol.identity().extractRotation(e.matrixWorld),this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(ol),this}intersectObject(e,t=!0,n=[]){return Ka(e,this,n,t),n.sort(ll),n}intersectObjects(e,t=!0,n=[]){for(let r=0,s=e.length;r<s;r++)Ka(e[r],this,n,t);return n.sort(ll),n}}function ll(i,e){return i.distance-e.distance}function Ka(i,e,t,n){let r=!0;if(i.layers.test(e.layers)&&i.raycast(e,t)===!1&&(r=!1),r===!0&&n===!0){const s=i.children;for(let a=0,o=s.length;a<o;a++)Ka(s[a],e,t,!0)}}class cl{constructor(e=1,t=0,n=0){this.radius=e,this.phi=t,this.theta=n}set(e,t,n){return this.radius=e,this.phi=t,this.theta=n,this}copy(e){return this.radius=e.radius,this.phi=e.phi,this.theta=e.theta,this}makeSafe(){return this.phi=nt(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(e){return this.setFromCartesianCoords(e.x,e.y,e.z)}setFromCartesianCoords(e,t,n){return this.radius=Math.sqrt(e*e+t*t+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(e,n),this.phi=Math.acos(nt(t/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}}class uu extends xi{constructor(e,t=null){super(),this.object=e,this.domElement=t,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(e){if(e===void 0){Xe("Controls: connect() now requires an element.");return}this.domElement!==null&&this.disconnect(),this.domElement=e}disconnect(){}dispose(){}update(){}}function hl(i,e,t,n){const r=du(n);switch(t){case Ql:return i*e;case gi:return i*e/r.components*r.byteLength;case lo:return i*e/r.components*r.byteLength;case mn:return i*e*2/r.components*r.byteLength;case co:return i*e*2/r.components*r.byteLength;case ec:return i*e*3/r.components*r.byteLength;case Ft:return i*e*4/r.components*r.byteLength;case ho:return i*e*4/r.components*r.byteLength;case as:case os:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case ls:case cs:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case va:case Ma:return Math.max(i,16)*Math.max(e,8)/4;case xa:case Sa:return Math.max(i,8)*Math.max(e,8)/2;case ya:case Ea:case Ta:case Aa:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*8;case ba:case wa:case Ca:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Ra:return Math.floor((i+3)/4)*Math.floor((e+3)/4)*16;case Pa:return Math.floor((i+4)/5)*Math.floor((e+3)/4)*16;case Da:return Math.floor((i+4)/5)*Math.floor((e+4)/5)*16;case Ia:return Math.floor((i+5)/6)*Math.floor((e+4)/5)*16;case La:return Math.floor((i+5)/6)*Math.floor((e+5)/6)*16;case Ua:return Math.floor((i+7)/8)*Math.floor((e+4)/5)*16;case Fa:return Math.floor((i+7)/8)*Math.floor((e+5)/6)*16;case Na:return Math.floor((i+7)/8)*Math.floor((e+7)/8)*16;case Oa:return Math.floor((i+9)/10)*Math.floor((e+4)/5)*16;case Ba:return Math.floor((i+9)/10)*Math.floor((e+5)/6)*16;case ka:return Math.floor((i+9)/10)*Math.floor((e+7)/8)*16;case za:return Math.floor((i+9)/10)*Math.floor((e+9)/10)*16;case Ga:return Math.floor((i+11)/12)*Math.floor((e+9)/10)*16;case Ha:return Math.floor((i+11)/12)*Math.floor((e+11)/12)*16;case Va:case Wa:case Xa:return Math.ceil(i/4)*Math.ceil(e/4)*16;case Ya:case qa:return Math.ceil(i/4)*Math.ceil(e/4)*8;case Za:case $a:return Math.ceil(i/4)*Math.ceil(e/4)*16}throw new Error(`Unable to determine texture byte length for ${t} format.`)}function du(i){switch(i){case Kt:case $l:return{byteLength:1,components:1};case mr:case jl:case Jt:return{byteLength:2,components:1};case ao:case oo:return{byteLength:2,components:4};case Cn:case so:case Yt:return{byteLength:4,components:1};case Kl:case Jl:return{byteLength:4,components:3}}throw new Error(`Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:io}}));typeof window<"u"&&(window.__THREE__?Xe("WARNING: Multiple instances of Three.js being imported."):window.__THREE__=io);function pc(){let i=null,e=!1,t=null,n=null;function r(s,a){t(s,a),n=i.requestAnimationFrame(r)}return{start:function(){e!==!0&&t!==null&&(n=i.requestAnimationFrame(r),e=!0)},stop:function(){i.cancelAnimationFrame(n),e=!1},setAnimationLoop:function(s){t=s},setContext:function(s){i=s}}}function fu(i){const e=new WeakMap;function t(o,c){const l=o.array,u=o.usage,d=l.byteLength,h=i.createBuffer();i.bindBuffer(c,h),i.bufferData(c,l,u),o.onUploadCallback();let f;if(l instanceof Float32Array)f=i.FLOAT;else if(typeof Float16Array<"u"&&l instanceof Float16Array)f=i.HALF_FLOAT;else if(l instanceof Uint16Array)o.isFloat16BufferAttribute?f=i.HALF_FLOAT:f=i.UNSIGNED_SHORT;else if(l instanceof Int16Array)f=i.SHORT;else if(l instanceof Uint32Array)f=i.UNSIGNED_INT;else if(l instanceof Int32Array)f=i.INT;else if(l instanceof Int8Array)f=i.BYTE;else if(l instanceof Uint8Array)f=i.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)f=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:h,type:f,bytesPerElement:l.BYTES_PER_ELEMENT,version:o.version,size:d}}function n(o,c,l){const u=c.array,d=c.updateRanges;if(i.bindBuffer(l,o),d.length===0)i.bufferSubData(l,0,u);else{d.sort((f,_)=>f.start-_.start);let h=0;for(let f=1;f<d.length;f++){const _=d[h],y=d[f];y.start<=_.start+_.count+1?_.count=Math.max(_.count,y.start+y.count-_.start):(++h,d[h]=y)}d.length=h+1;for(let f=0,_=d.length;f<_;f++){const y=d[f];i.bufferSubData(l,y.start*u.BYTES_PER_ELEMENT,u,y.start,y.count)}c.clearUpdateRanges()}c.onUploadCallback()}function r(o){return o.isInterleavedBufferAttribute&&(o=o.data),e.get(o)}function s(o){o.isInterleavedBufferAttribute&&(o=o.data);const c=e.get(o);c&&(i.deleteBuffer(c.buffer),e.delete(o))}function a(o,c){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){const u=e.get(o);(!u||u.version<o.version)&&e.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}const l=e.get(o);if(l===void 0)e.set(o,t(o,c));else if(l.version<o.version){if(l.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(l.buffer,o,c),l.version=o.version}}return{get:r,remove:s,update:a}}var pu=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,mu=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,gu=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,_u=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,xu=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,vu=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Su=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Mu=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,yu=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,Eu=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,bu=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Tu=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Au=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,wu=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Cu=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Ru=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,Pu=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Du=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,Iu=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Lu=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,Uu=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,Fu=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,Nu=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,Ou=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
vec3 inverseTransformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( vec4( dir, 0.0 ) * matrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,Bu=`#ifdef ENVMAP_TYPE_CUBE_UV
	#define cubeUV_minMipLevel 4.0
	#define cubeUV_minTileSize 16.0
	float getFace( vec3 direction ) {
		vec3 absDirection = abs( direction );
		float face = - 1.0;
		if ( absDirection.x > absDirection.z ) {
			if ( absDirection.x > absDirection.y )
				face = direction.x > 0.0 ? 0.0 : 3.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		} else {
			if ( absDirection.z > absDirection.y )
				face = direction.z > 0.0 ? 2.0 : 5.0;
			else
				face = direction.y > 0.0 ? 1.0 : 4.0;
		}
		return face;
	}
	vec2 getUV( vec3 direction, float face ) {
		vec2 uv;
		if ( face == 0.0 ) {
			uv = vec2( direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 1.0 ) {
			uv = vec2( - direction.x, - direction.z ) / abs( direction.y );
		} else if ( face == 2.0 ) {
			uv = vec2( - direction.x, direction.y ) / abs( direction.z );
		} else if ( face == 3.0 ) {
			uv = vec2( - direction.z, direction.y ) / abs( direction.x );
		} else if ( face == 4.0 ) {
			uv = vec2( - direction.x, direction.z ) / abs( direction.y );
		} else {
			uv = vec2( direction.x, direction.y ) / abs( direction.z );
		}
		return 0.5 * ( uv + 1.0 );
	}
	vec3 bilinearCubeUV( sampler2D envMap, vec3 direction, float mipInt ) {
		float face = getFace( direction );
		float filterInt = max( cubeUV_minMipLevel - mipInt, 0.0 );
		mipInt = max( mipInt, cubeUV_minMipLevel );
		float faceSize = exp2( mipInt );
		highp vec2 uv = getUV( direction, face ) * ( faceSize - 2.0 ) + 1.0;
		if ( face > 2.0 ) {
			uv.y += faceSize;
			face -= 3.0;
		}
		uv.x += face * faceSize;
		uv.x += filterInt * 3.0 * cubeUV_minTileSize;
		uv.y += 4.0 * ( exp2( CUBEUV_MAX_MIP ) - faceSize );
		uv.x *= CUBEUV_TEXEL_WIDTH;
		uv.y *= CUBEUV_TEXEL_HEIGHT;
		#ifdef texture2DGradEXT
			return texture2DGradEXT( envMap, uv, vec2( 0.0 ), vec2( 0.0 ) ).rgb;
		#else
			return texture2D( envMap, uv ).rgb;
		#endif
	}
	#define cubeUV_r0 1.0
	#define cubeUV_m0 - 2.0
	#define cubeUV_r1 0.8
	#define cubeUV_m1 - 1.0
	#define cubeUV_r4 0.4
	#define cubeUV_m4 2.0
	#define cubeUV_r5 0.305
	#define cubeUV_m5 3.0
	#define cubeUV_r6 0.21
	#define cubeUV_m6 4.0
	float roughnessToMip( float roughness ) {
		float mip = 0.0;
		if ( roughness >= cubeUV_r1 ) {
			mip = ( cubeUV_r0 - roughness ) * ( cubeUV_m1 - cubeUV_m0 ) / ( cubeUV_r0 - cubeUV_r1 ) + cubeUV_m0;
		} else if ( roughness >= cubeUV_r4 ) {
			mip = ( cubeUV_r1 - roughness ) * ( cubeUV_m4 - cubeUV_m1 ) / ( cubeUV_r1 - cubeUV_r4 ) + cubeUV_m1;
		} else if ( roughness >= cubeUV_r5 ) {
			mip = ( cubeUV_r4 - roughness ) * ( cubeUV_m5 - cubeUV_m4 ) / ( cubeUV_r4 - cubeUV_r5 ) + cubeUV_m4;
		} else if ( roughness >= cubeUV_r6 ) {
			mip = ( cubeUV_r5 - roughness ) * ( cubeUV_m6 - cubeUV_m5 ) / ( cubeUV_r5 - cubeUV_r6 ) + cubeUV_m5;
		} else {
			mip = - 2.0 * log2( 1.16 * roughness );		}
		return mip;
	}
	vec4 textureCubeUV( sampler2D envMap, vec3 sampleDir, float roughness ) {
		float mip = clamp( roughnessToMip( roughness ), cubeUV_m0, CUBEUV_MAX_MIP );
		float mipF = fract( mip );
		float mipInt = floor( mip );
		vec3 color0 = bilinearCubeUV( envMap, sampleDir, mipInt );
		if ( mipF == 0.0 ) {
			return vec4( color0, 1.0 );
		} else {
			vec3 color1 = bilinearCubeUV( envMap, sampleDir, mipInt + 1.0 );
			return vec4( mix( color0, color1, mipF ), 1.0 );
		}
	}
#endif`,ku=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
	#ifdef FLIP_SIDED
		transformedTangent = - transformedTangent;
	#endif
#endif`,zu=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,Gu=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,Hu=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,Vu=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,Wu="gl_FragColor = linearToOutputTexel( gl_FragColor );",Xu=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,Yu=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,qu=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform float flipEnvMap;
	uniform mat3 envMapRotation;
	#ifdef ENVMAP_TYPE_CUBE
		uniform samplerCube envMap;
	#else
		uniform sampler2D envMap;
	#endif
#endif`,Zu=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,$u=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,ju=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,Ku=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,Ju=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,Qu=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,ed=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,td=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,nd=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,id=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,rd=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,sd=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif`,ad=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 worldNormal = inverseTransformDirection( normal, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * worldNormal, 1.0 );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_CUBE_UV
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = inverseTransformDirection( reflectVec, viewMatrix );
			vec4 envMapColor = textureCubeUV( envMap, envMapRotation * reflectVec, roughness );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_CUBE_UV
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
#endif`,od=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,ld=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,cd=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,hd=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,ud=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.0525 );material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.0525 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,dd=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0;
		vec3 iridescenceFresnelDielectric;
		vec3 iridescenceFresnelMetallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		float v = 0.5 / ( gv + gl );
		return v;
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = material.clearcoatRoughness;
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = material.roughness;
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float V = V_GGX_SmithCorrelated_Anisotropic( material.alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( material.alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
vec3 BRDF_GGX_Multiscatter( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 singleScatter = BRDF_GGX( lightDir, viewDir, normal, material );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 dfgV = texture2D( dfgLUT, vec2( material.roughness, dotNV ) ).rg;
	vec2 dfgL = texture2D( dfgLUT, vec2( material.roughness, dotNL ) ).rg;
	vec3 FssEss_V = material.specularColorBlended * dfgV.x + material.specularF90 * dfgV.y;
	vec3 FssEss_L = material.specularColorBlended * dfgL.x + material.specularF90 * dfgL.y;
	float Ess_V = dfgV.x + dfgV.y;
	float Ess_L = dfgL.x + dfgL.y;
	float Ems_V = 1.0 - Ess_V;
	float Ems_L = 1.0 - Ess_L;
	vec3 Favg = material.specularColorBlended + ( 1.0 - material.specularColorBlended ) * 0.047619;
	vec3 Fms = FssEss_V * FssEss_L * Favg / ( 1.0 - Ems_V * Ems_L * Favg + EPSILON );
	float compensationFactor = Ems_V * Ems_L;
	vec3 multiScatter = Fms * compensationFactor;
	return singleScatter + multiScatter;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	reflectedLight.directSpecular += irradiance * BRDF_GGX_Multiscatter( directLight.direction, geometryViewDir, geometryNormal, material );
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseContribution );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution );
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.iridescence, material.iridescenceFresnelDielectric, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceFresnelMetallic, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( geometryNormal, geometryViewDir, material.specularColor, material.specularF90, material.roughness, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( geometryNormal, geometryViewDir, material.diffuseColor, material.specularF90, material.roughness, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,fd=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		material.iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		material.iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( material.iridescenceFresnelDielectric, material.iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0 = Schlick_to_F0( material.iridescenceFresnel, 1.0, dotNVi );
	}
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,pd=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_CUBE_UV )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		radiance += getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		radiance += getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,md=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,gd=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,_d=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,xd=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,vd=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Sd=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Md=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,yd=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,Ed=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,bd=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Td=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Ad=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,wd=`#if defined( USE_MORPHCOLORS )
	vColor *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Cd=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Rd=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,Pd=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Dd=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#if defined( DOUBLE_SIDED ) && ! defined( FLAT_SHADED )
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,Id=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Ld=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Ud=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Fd=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
	#endif
#endif`,Nd=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Od=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,Bd=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,kd=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,zd=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,Gd=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,Hd=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,Vd=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,Wd=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,Xd=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,Yd=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,qd=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,Zd=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,$d=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			shadowCoord.z += shadowBias;
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				float radius = shadowRadius * texelSize.x;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float dp = ( shadowCameraNear * ( shadowCameraFar - viewSpaceZ ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp -= shadowBias;
			#else
				float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
				dp += shadowBias;
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			vec2 sample0 = vogelDiskSample( 0, 5, phi );
			vec2 sample1 = vogelDiskSample( 1, 5, phi );
			vec2 sample2 = vogelDiskSample( 2, 5, phi );
			vec2 sample3 = vogelDiskSample( 3, 5, phi );
			vec2 sample4 = vogelDiskSample( 4, 5, phi );
			shadow = (
				texture( shadowMap, vec4( bd3D + ( tangent * sample0.x + bitangent * sample0.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample1.x + bitangent * sample1.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample2.x + bitangent * sample2.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample3.x + bitangent * sample3.y ) * texelSize, dp ) ) +
				texture( shadowMap, vec4( bd3D + ( tangent * sample4.x + bitangent * sample4.y ) * texelSize, dp ) )
			) * 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,jd=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,Kd=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	vec3 shadowWorldNormal = inverseTransformDirection( transformedNormal, viewMatrix );
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,Jd=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,Qd=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,ef=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,tf=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,nf=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,rf=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,sf=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,af=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,of=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,lf=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = inverseTransformDirection( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,cf=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,hf=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,uf=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,df=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,ff=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`;const pf=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,mf=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,gf=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,_f=`#ifdef ENVMAP_TYPE_CUBE
	uniform samplerCube envMap;
#elif defined( ENVMAP_TYPE_CUBE_UV )
	uniform sampler2D envMap;
#endif
uniform float flipEnvMap;
uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <cube_uv_reflection_fragment>
void main() {
	#ifdef ENVMAP_TYPE_CUBE
		vec4 texColor = textureCube( envMap, backgroundRotation * vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) );
	#elif defined( ENVMAP_TYPE_CUBE_UV )
		vec4 texColor = textureCubeUV( envMap, backgroundRotation * vWorldDirection, backgroundBlurriness );
	#else
		vec4 texColor = vec4( 0.0, 0.0, 0.0, 1.0 );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,xf=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,vf=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Sf=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Mf=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,yf=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Ef=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main () {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,bf=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,Tf=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Af=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,wf=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Cf=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Rf=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Pf=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Df=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,If=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,Lf=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Uf=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Ff=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Nf=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Of=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Bf=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,kf=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <cube_uv_reflection_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,zf=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Gf=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Hf=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,Vf=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Wf=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,Xf=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Yf=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,qf=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,Qe={alphahash_fragment:pu,alphahash_pars_fragment:mu,alphamap_fragment:gu,alphamap_pars_fragment:_u,alphatest_fragment:xu,alphatest_pars_fragment:vu,aomap_fragment:Su,aomap_pars_fragment:Mu,batching_pars_vertex:yu,batching_vertex:Eu,begin_vertex:bu,beginnormal_vertex:Tu,bsdfs:Au,iridescence_fragment:wu,bumpmap_pars_fragment:Cu,clipping_planes_fragment:Ru,clipping_planes_pars_fragment:Pu,clipping_planes_pars_vertex:Du,clipping_planes_vertex:Iu,color_fragment:Lu,color_pars_fragment:Uu,color_pars_vertex:Fu,color_vertex:Nu,common:Ou,cube_uv_reflection_fragment:Bu,defaultnormal_vertex:ku,displacementmap_pars_vertex:zu,displacementmap_vertex:Gu,emissivemap_fragment:Hu,emissivemap_pars_fragment:Vu,colorspace_fragment:Wu,colorspace_pars_fragment:Xu,envmap_fragment:Yu,envmap_common_pars_fragment:qu,envmap_pars_fragment:Zu,envmap_pars_vertex:$u,envmap_physical_pars_fragment:ad,envmap_vertex:ju,fog_vertex:Ku,fog_pars_vertex:Ju,fog_fragment:Qu,fog_pars_fragment:ed,gradientmap_pars_fragment:td,lightmap_pars_fragment:nd,lights_lambert_fragment:id,lights_lambert_pars_fragment:rd,lights_pars_begin:sd,lights_toon_fragment:od,lights_toon_pars_fragment:ld,lights_phong_fragment:cd,lights_phong_pars_fragment:hd,lights_physical_fragment:ud,lights_physical_pars_fragment:dd,lights_fragment_begin:fd,lights_fragment_maps:pd,lights_fragment_end:md,logdepthbuf_fragment:gd,logdepthbuf_pars_fragment:_d,logdepthbuf_pars_vertex:xd,logdepthbuf_vertex:vd,map_fragment:Sd,map_pars_fragment:Md,map_particle_fragment:yd,map_particle_pars_fragment:Ed,metalnessmap_fragment:bd,metalnessmap_pars_fragment:Td,morphinstance_vertex:Ad,morphcolor_vertex:wd,morphnormal_vertex:Cd,morphtarget_pars_vertex:Rd,morphtarget_vertex:Pd,normal_fragment_begin:Dd,normal_fragment_maps:Id,normal_pars_fragment:Ld,normal_pars_vertex:Ud,normal_vertex:Fd,normalmap_pars_fragment:Nd,clearcoat_normal_fragment_begin:Od,clearcoat_normal_fragment_maps:Bd,clearcoat_pars_fragment:kd,iridescence_pars_fragment:zd,opaque_fragment:Gd,packing:Hd,premultiplied_alpha_fragment:Vd,project_vertex:Wd,dithering_fragment:Xd,dithering_pars_fragment:Yd,roughnessmap_fragment:qd,roughnessmap_pars_fragment:Zd,shadowmap_pars_fragment:$d,shadowmap_pars_vertex:jd,shadowmap_vertex:Kd,shadowmask_pars_fragment:Jd,skinbase_vertex:Qd,skinning_pars_vertex:ef,skinning_vertex:tf,skinnormal_vertex:nf,specularmap_fragment:rf,specularmap_pars_fragment:sf,tonemapping_fragment:af,tonemapping_pars_fragment:of,transmission_fragment:lf,transmission_pars_fragment:cf,uv_pars_fragment:hf,uv_pars_vertex:uf,uv_vertex:df,worldpos_vertex:ff,background_vert:pf,background_frag:mf,backgroundCube_vert:gf,backgroundCube_frag:_f,cube_vert:xf,cube_frag:vf,depth_vert:Sf,depth_frag:Mf,distance_vert:yf,distance_frag:Ef,equirect_vert:bf,equirect_frag:Tf,linedashed_vert:Af,linedashed_frag:wf,meshbasic_vert:Cf,meshbasic_frag:Rf,meshlambert_vert:Pf,meshlambert_frag:Df,meshmatcap_vert:If,meshmatcap_frag:Lf,meshnormal_vert:Uf,meshnormal_frag:Ff,meshphong_vert:Nf,meshphong_frag:Of,meshphysical_vert:Bf,meshphysical_frag:kf,meshtoon_vert:zf,meshtoon_frag:Gf,points_vert:Hf,points_frag:Vf,shadow_vert:Wf,shadow_frag:Xf,sprite_vert:Yf,sprite_frag:qf},be={common:{diffuse:{value:new rt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Je},alphaMap:{value:null},alphaMapTransform:{value:new Je},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Je}},envmap:{envMap:{value:null},envMapRotation:{value:new Je},flipEnvMap:{value:-1},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Je}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Je}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Je},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Je},normalScale:{value:new $e(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Je},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Je}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Je}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Je}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new rt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null}},points:{diffuse:{value:new rt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Je},alphaTest:{value:0},uvTransform:{value:new Je}},sprite:{diffuse:{value:new rt(16777215)},opacity:{value:1},center:{value:new $e(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Je},alphaMap:{value:null},alphaMapTransform:{value:new Je},alphaTest:{value:0}}},bn={basic:{uniforms:zt([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.fog]),vertexShader:Qe.meshbasic_vert,fragmentShader:Qe.meshbasic_frag},lambert:{uniforms:zt([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.fog,be.lights,{emissive:{value:new rt(0)},envMapIntensity:{value:1}}]),vertexShader:Qe.meshlambert_vert,fragmentShader:Qe.meshlambert_frag},phong:{uniforms:zt([be.common,be.specularmap,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.fog,be.lights,{emissive:{value:new rt(0)},specular:{value:new rt(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:Qe.meshphong_vert,fragmentShader:Qe.meshphong_frag},standard:{uniforms:zt([be.common,be.envmap,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.roughnessmap,be.metalnessmap,be.fog,be.lights,{emissive:{value:new rt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:Qe.meshphysical_vert,fragmentShader:Qe.meshphysical_frag},toon:{uniforms:zt([be.common,be.aomap,be.lightmap,be.emissivemap,be.bumpmap,be.normalmap,be.displacementmap,be.gradientmap,be.fog,be.lights,{emissive:{value:new rt(0)}}]),vertexShader:Qe.meshtoon_vert,fragmentShader:Qe.meshtoon_frag},matcap:{uniforms:zt([be.common,be.bumpmap,be.normalmap,be.displacementmap,be.fog,{matcap:{value:null}}]),vertexShader:Qe.meshmatcap_vert,fragmentShader:Qe.meshmatcap_frag},points:{uniforms:zt([be.points,be.fog]),vertexShader:Qe.points_vert,fragmentShader:Qe.points_frag},dashed:{uniforms:zt([be.common,be.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:Qe.linedashed_vert,fragmentShader:Qe.linedashed_frag},depth:{uniforms:zt([be.common,be.displacementmap]),vertexShader:Qe.depth_vert,fragmentShader:Qe.depth_frag},normal:{uniforms:zt([be.common,be.bumpmap,be.normalmap,be.displacementmap,{opacity:{value:1}}]),vertexShader:Qe.meshnormal_vert,fragmentShader:Qe.meshnormal_frag},sprite:{uniforms:zt([be.sprite,be.fog]),vertexShader:Qe.sprite_vert,fragmentShader:Qe.sprite_frag},background:{uniforms:{uvTransform:{value:new Je},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:Qe.background_vert,fragmentShader:Qe.background_frag},backgroundCube:{uniforms:{envMap:{value:null},flipEnvMap:{value:-1},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Je}},vertexShader:Qe.backgroundCube_vert,fragmentShader:Qe.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:Qe.cube_vert,fragmentShader:Qe.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:Qe.equirect_vert,fragmentShader:Qe.equirect_frag},distance:{uniforms:zt([be.common,be.displacementmap,{referencePosition:{value:new q},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:Qe.distance_vert,fragmentShader:Qe.distance_frag},shadow:{uniforms:zt([be.lights,be.fog,{color:{value:new rt(0)},opacity:{value:1}}]),vertexShader:Qe.shadow_vert,fragmentShader:Qe.shadow_frag}};bn.physical={uniforms:zt([bn.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Je},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Je},clearcoatNormalScale:{value:new $e(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Je},dispersion:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Je},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Je},sheen:{value:0},sheenColor:{value:new rt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Je},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Je},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Je},transmissionSamplerSize:{value:new $e},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Je},attenuationDistance:{value:0},attenuationColor:{value:new rt(0)},specularColor:{value:new rt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Je},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Je},anisotropyVector:{value:new $e},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Je}}]),vertexShader:Qe.meshphysical_vert,fragmentShader:Qe.meshphysical_frag};const es={r:0,b:0,g:0},ui=new Rn,Zf=new _t;function $f(i,e,t,n,r,s){const a=new rt(0);let o=r===!0?0:1,c,l,u=null,d=0,h=null;function f(b){let w=b.isScene===!0?b.background:null;if(w&&w.isTexture){const A=b.backgroundBlurriness>0;w=e.get(w,A)}return w}function _(b){let w=!1;const A=f(b);A===null?g(a,o):A&&A.isColor&&(g(A,1),w=!0);const U=i.xr.getEnvironmentBlendMode();U==="additive"?t.buffers.color.setClear(0,0,0,1,s):U==="alpha-blend"&&t.buffers.color.setClear(0,0,0,0,s),(i.autoClear||w)&&(t.buffers.depth.setTest(!0),t.buffers.depth.setMask(!0),t.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function y(b,w){const A=f(w);A&&(A.isCubeTexture||A.mapping===gs)?(l===void 0&&(l=new _n(new ii(1,1,1),new Pn({name:"BackgroundCubeMaterial",uniforms:Xi(bn.backgroundCube.uniforms),vertexShader:bn.backgroundCube.vertexShader,fragmentShader:bn.backgroundCube.fragmentShader,side:qt,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),l.geometry.deleteAttribute("uv"),l.onBeforeRender=function(U,L,N){this.matrixWorld.copyPosition(N.matrixWorld)},Object.defineProperty(l.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),n.update(l)),ui.copy(w.backgroundRotation),ui.x*=-1,ui.y*=-1,ui.z*=-1,A.isCubeTexture&&A.isRenderTargetTexture===!1&&(ui.y*=-1,ui.z*=-1),l.material.uniforms.envMap.value=A,l.material.uniforms.flipEnvMap.value=A.isCubeTexture&&A.isRenderTargetTexture===!1?-1:1,l.material.uniforms.backgroundBlurriness.value=w.backgroundBlurriness,l.material.uniforms.backgroundIntensity.value=w.backgroundIntensity,l.material.uniforms.backgroundRotation.value.setFromMatrix4(Zf.makeRotationFromEuler(ui)),l.material.toneMapped=lt.getTransfer(A.colorSpace)!==ft,(u!==A||d!==A.version||h!==i.toneMapping)&&(l.material.needsUpdate=!0,u=A,d=A.version,h=i.toneMapping),l.layers.enableAll(),b.unshift(l,l.geometry,l.material,0,0,null)):A&&A.isTexture&&(c===void 0&&(c=new _n(new xs(2,2),new Pn({name:"BackgroundMaterial",uniforms:Xi(bn.background.uniforms),vertexShader:bn.background.vertexShader,fragmentShader:bn.background.fragmentShader,side:ri,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),n.update(c)),c.material.uniforms.t2D.value=A,c.material.uniforms.backgroundIntensity.value=w.backgroundIntensity,c.material.toneMapped=lt.getTransfer(A.colorSpace)!==ft,A.matrixAutoUpdate===!0&&A.updateMatrix(),c.material.uniforms.uvTransform.value.copy(A.matrix),(u!==A||d!==A.version||h!==i.toneMapping)&&(c.material.needsUpdate=!0,u=A,d=A.version,h=i.toneMapping),c.layers.enableAll(),b.unshift(c,c.geometry,c.material,0,0,null))}function g(b,w){b.getRGB(es,uc(i)),t.buffers.color.setClear(es.r,es.g,es.b,w,s)}function m(){l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(b,w=1){a.set(b),o=w,g(a,o)},getClearAlpha:function(){return o},setClearAlpha:function(b){o=b,g(a,o)},render:_,addToRenderList:y,dispose:m}}function jf(i,e){const t=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},r=h(null);let s=r,a=!1;function o(D,O,V,K,Y){let Z=!1;const X=d(D,K,V,O);s!==X&&(s=X,l(s.object)),Z=f(D,K,V,Y),Z&&_(D,K,V,Y),Y!==null&&e.update(Y,i.ELEMENT_ARRAY_BUFFER),(Z||a)&&(a=!1,A(D,O,V,K),Y!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,e.get(Y).buffer))}function c(){return i.createVertexArray()}function l(D){return i.bindVertexArray(D)}function u(D){return i.deleteVertexArray(D)}function d(D,O,V,K){const Y=K.wireframe===!0;let Z=n[O.id];Z===void 0&&(Z={},n[O.id]=Z);const X=D.isInstancedMesh===!0?D.id:0;let fe=Z[X];fe===void 0&&(fe={},Z[X]=fe);let oe=fe[V.id];oe===void 0&&(oe={},fe[V.id]=oe);let ye=oe[Y];return ye===void 0&&(ye=h(c()),oe[Y]=ye),ye}function h(D){const O=[],V=[],K=[];for(let Y=0;Y<t;Y++)O[Y]=0,V[Y]=0,K[Y]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:O,enabledAttributes:V,attributeDivisors:K,object:D,attributes:{},index:null}}function f(D,O,V,K){const Y=s.attributes,Z=O.attributes;let X=0;const fe=V.getAttributes();for(const oe in fe)if(fe[oe].location>=0){const Ae=Y[oe];let ve=Z[oe];if(ve===void 0&&(oe==="instanceMatrix"&&D.instanceMatrix&&(ve=D.instanceMatrix),oe==="instanceColor"&&D.instanceColor&&(ve=D.instanceColor)),Ae===void 0||Ae.attribute!==ve||ve&&Ae.data!==ve.data)return!0;X++}return s.attributesNum!==X||s.index!==K}function _(D,O,V,K){const Y={},Z=O.attributes;let X=0;const fe=V.getAttributes();for(const oe in fe)if(fe[oe].location>=0){let Ae=Z[oe];Ae===void 0&&(oe==="instanceMatrix"&&D.instanceMatrix&&(Ae=D.instanceMatrix),oe==="instanceColor"&&D.instanceColor&&(Ae=D.instanceColor));const ve={};ve.attribute=Ae,Ae&&Ae.data&&(ve.data=Ae.data),Y[oe]=ve,X++}s.attributes=Y,s.attributesNum=X,s.index=K}function y(){const D=s.newAttributes;for(let O=0,V=D.length;O<V;O++)D[O]=0}function g(D){m(D,0)}function m(D,O){const V=s.newAttributes,K=s.enabledAttributes,Y=s.attributeDivisors;V[D]=1,K[D]===0&&(i.enableVertexAttribArray(D),K[D]=1),Y[D]!==O&&(i.vertexAttribDivisor(D,O),Y[D]=O)}function b(){const D=s.newAttributes,O=s.enabledAttributes;for(let V=0,K=O.length;V<K;V++)O[V]!==D[V]&&(i.disableVertexAttribArray(V),O[V]=0)}function w(D,O,V,K,Y,Z,X){X===!0?i.vertexAttribIPointer(D,O,V,Y,Z):i.vertexAttribPointer(D,O,V,K,Y,Z)}function A(D,O,V,K){y();const Y=K.attributes,Z=V.getAttributes(),X=O.defaultAttributeValues;for(const fe in Z){const oe=Z[fe];if(oe.location>=0){let ye=Y[fe];if(ye===void 0&&(fe==="instanceMatrix"&&D.instanceMatrix&&(ye=D.instanceMatrix),fe==="instanceColor"&&D.instanceColor&&(ye=D.instanceColor)),ye!==void 0){const Ae=ye.normalized,ve=ye.itemSize,Ge=e.get(ye);if(Ge===void 0)continue;const st=Ge.buffer,_e=Ge.type,$=Ge.bytesPerElement,ue=_e===i.INT||_e===i.UNSIGNED_INT||ye.gpuType===so;if(ye.isInterleavedBufferAttribute){const de=ye.data,ze=de.stride,Le=ye.offset;if(de.isInstancedInterleavedBuffer){for(let Fe=0;Fe<oe.locationSize;Fe++)m(oe.location+Fe,de.meshPerAttribute);D.isInstancedMesh!==!0&&K._maxInstanceCount===void 0&&(K._maxInstanceCount=de.meshPerAttribute*de.count)}else for(let Fe=0;Fe<oe.locationSize;Fe++)g(oe.location+Fe);i.bindBuffer(i.ARRAY_BUFFER,st);for(let Fe=0;Fe<oe.locationSize;Fe++)w(oe.location+Fe,ve/oe.locationSize,_e,Ae,ze*$,(Le+ve/oe.locationSize*Fe)*$,ue)}else{if(ye.isInstancedBufferAttribute){for(let de=0;de<oe.locationSize;de++)m(oe.location+de,ye.meshPerAttribute);D.isInstancedMesh!==!0&&K._maxInstanceCount===void 0&&(K._maxInstanceCount=ye.meshPerAttribute*ye.count)}else for(let de=0;de<oe.locationSize;de++)g(oe.location+de);i.bindBuffer(i.ARRAY_BUFFER,st);for(let de=0;de<oe.locationSize;de++)w(oe.location+de,ve/oe.locationSize,_e,Ae,ve*$,ve/oe.locationSize*de*$,ue)}}else if(X!==void 0){const Ae=X[fe];if(Ae!==void 0)switch(Ae.length){case 2:i.vertexAttrib2fv(oe.location,Ae);break;case 3:i.vertexAttrib3fv(oe.location,Ae);break;case 4:i.vertexAttrib4fv(oe.location,Ae);break;default:i.vertexAttrib1fv(oe.location,Ae)}}}}b()}function U(){T();for(const D in n){const O=n[D];for(const V in O){const K=O[V];for(const Y in K){const Z=K[Y];for(const X in Z)u(Z[X].object),delete Z[X];delete K[Y]}}delete n[D]}}function L(D){if(n[D.id]===void 0)return;const O=n[D.id];for(const V in O){const K=O[V];for(const Y in K){const Z=K[Y];for(const X in Z)u(Z[X].object),delete Z[X];delete K[Y]}}delete n[D.id]}function N(D){for(const O in n){const V=n[O];for(const K in V){const Y=V[K];if(Y[D.id]===void 0)continue;const Z=Y[D.id];for(const X in Z)u(Z[X].object),delete Z[X];delete Y[D.id]}}}function S(D){for(const O in n){const V=n[O],K=D.isInstancedMesh===!0?D.id:0,Y=V[K];if(Y!==void 0){for(const Z in Y){const X=Y[Z];for(const fe in X)u(X[fe].object),delete X[fe];delete Y[Z]}delete V[K],Object.keys(V).length===0&&delete n[O]}}}function T(){G(),a=!0,s!==r&&(s=r,l(s.object))}function G(){r.geometry=null,r.program=null,r.wireframe=!1}return{setup:o,reset:T,resetDefaultState:G,dispose:U,releaseStatesOfGeometry:L,releaseStatesOfObject:S,releaseStatesOfProgram:N,initAttributes:y,enableAttribute:g,disableUnusedAttributes:b}}function Kf(i,e,t){let n;function r(l){n=l}function s(l,u){i.drawArrays(n,l,u),t.update(u,n,1)}function a(l,u,d){d!==0&&(i.drawArraysInstanced(n,l,u,d),t.update(u,n,d))}function o(l,u,d){if(d===0)return;e.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,l,0,u,0,d);let f=0;for(let _=0;_<d;_++)f+=u[_];t.update(f,n,1)}function c(l,u,d,h){if(d===0)return;const f=e.get("WEBGL_multi_draw");if(f===null)for(let _=0;_<l.length;_++)a(l[_],u[_],h[_]);else{f.multiDrawArraysInstancedWEBGL(n,l,0,u,0,h,0,d);let _=0;for(let y=0;y<d;y++)_+=u[y]*h[y];t.update(_,n,1)}}this.setMode=r,this.render=s,this.renderInstances=a,this.renderMultiDraw=o,this.renderMultiDrawInstances=c}function Jf(i,e,t,n){let r;function s(){if(r!==void 0)return r;if(e.has("EXT_texture_filter_anisotropic")===!0){const N=e.get("EXT_texture_filter_anisotropic");r=i.getParameter(N.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else r=0;return r}function a(N){return!(N!==Ft&&n.convert(N)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(N){const S=N===Jt&&(e.has("EXT_color_buffer_half_float")||e.has("EXT_color_buffer_float"));return!(N!==Kt&&n.convert(N)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE)&&N!==Yt&&!S)}function c(N){if(N==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";N="mediump"}return N==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=t.precision!==void 0?t.precision:"highp";const u=c(l);u!==l&&(Xe("WebGLRenderer:",l,"not supported, using",u,"instead."),l=u);const d=t.logarithmicDepthBuffer===!0,h=t.reversedDepthBuffer===!0&&e.has("EXT_clip_control"),f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),_=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),y=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),m=i.getParameter(i.MAX_VERTEX_ATTRIBS),b=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),w=i.getParameter(i.MAX_VARYING_VECTORS),A=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),U=i.getParameter(i.MAX_SAMPLES),L=i.getParameter(i.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:s,getMaxPrecision:c,textureFormatReadable:a,textureTypeReadable:o,precision:l,logarithmicDepthBuffer:d,reversedDepthBuffer:h,maxTextures:f,maxVertexTextures:_,maxTextureSize:y,maxCubemapSize:g,maxAttributes:m,maxVertexUniforms:b,maxVaryings:w,maxFragmentUniforms:A,maxSamples:U,samples:L}}function Qf(i){const e=this;let t=null,n=0,r=!1,s=!1;const a=new Qn,o=new Je,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(d,h){const f=d.length!==0||h||n!==0||r;return r=h,n=d.length,f},this.beginShadows=function(){s=!0,u(null)},this.endShadows=function(){s=!1},this.setGlobalState=function(d,h){t=u(d,h,0)},this.setState=function(d,h,f){const _=d.clippingPlanes,y=d.clipIntersection,g=d.clipShadows,m=i.get(d);if(!r||_===null||_.length===0||s&&!g)s?u(null):l();else{const b=s?0:n,w=b*4;let A=m.clippingState||null;c.value=A,A=u(_,h,w,f);for(let U=0;U!==w;++U)A[U]=t[U];m.clippingState=A,this.numIntersection=y?this.numPlanes:0,this.numPlanes+=b}};function l(){c.value!==t&&(c.value=t,c.needsUpdate=n>0),e.numPlanes=n,e.numIntersection=0}function u(d,h,f,_){const y=d!==null?d.length:0;let g=null;if(y!==0){if(g=c.value,_!==!0||g===null){const m=f+y*4,b=h.matrixWorldInverse;o.getNormalMatrix(b),(g===null||g.length<m)&&(g=new Float32Array(m));for(let w=0,A=f;w!==y;++w,A+=4)a.copy(d[w]).applyMatrix4(b,o),a.normal.toArray(g,A),g[A+3]=a.constant}c.value=g,c.needsUpdate=!0}return e.numPlanes=y,e.numIntersection=0,g}}const ni=4,ul=[.125,.215,.35,.446,.526,.582],pi=20,ep=256,ar=new Mo,dl=new rt;let js=null,Ks=0,Js=0,Qs=!1;const tp=new q;class fl{constructor(e){this._renderer=e,this._pingPongRenderTarget=null,this._lodMax=0,this._cubeSize=0,this._sizeLods=[],this._sigmas=[],this._lodMeshes=[],this._backgroundBox=null,this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null}fromScene(e,t=0,n=.1,r=100,s={}){const{size:a=256,position:o=tp}=s;js=this._renderer.getRenderTarget(),Ks=this._renderer.getActiveCubeFace(),Js=this._renderer.getActiveMipmapLevel(),Qs=this._renderer.xr.enabled,this._renderer.xr.enabled=!1,this._setSize(a);const c=this._allocateTargets();return c.depthBuffer=!0,this._sceneToCubeUV(e,n,r,c,o),t>0&&this._blur(c,0,0,t),this._applyPMREM(c),this._cleanup(c),c}fromEquirectangular(e,t=null){return this._fromTexture(e,t)}fromCubemap(e,t=null){return this._fromTexture(e,t)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=gl(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=ml(),this._compileMaterial(this._equirectMaterial))}dispose(){this._dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._backgroundBox!==null&&(this._backgroundBox.geometry.dispose(),this._backgroundBox.material.dispose())}_setSize(e){this._lodMax=Math.floor(Math.log2(e)),this._cubeSize=Math.pow(2,this._lodMax)}_dispose(){this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._pingPongRenderTarget!==null&&this._pingPongRenderTarget.dispose();for(let e=0;e<this._lodMeshes.length;e++)this._lodMeshes[e].geometry.dispose()}_cleanup(e){this._renderer.setRenderTarget(js,Ks,Js),this._renderer.xr.enabled=Qs,e.scissorTest=!1,Ni(e,0,0,e.width,e.height)}_fromTexture(e,t){e.mapping===_i||e.mapping===Wi?this._setSize(e.image.length===0?16:e.image[0].width||e.image[0].image.width):this._setSize(e.image.width/4),js=this._renderer.getRenderTarget(),Ks=this._renderer.getActiveCubeFace(),Js=this._renderer.getActiveMipmapLevel(),Qs=this._renderer.xr.enabled,this._renderer.xr.enabled=!1;const n=t||this._allocateTargets();return this._textureToCubeUV(e,n),this._applyPMREM(n),this._cleanup(n),n}_allocateTargets(){const e=3*Math.max(this._cubeSize,112),t=4*this._cubeSize,n={magFilter:bt,minFilter:bt,generateMipmaps:!1,type:Jt,format:Ft,colorSpace:jt,depthBuffer:!1},r=pl(e,t,n);if(this._pingPongRenderTarget===null||this._pingPongRenderTarget.width!==e||this._pingPongRenderTarget.height!==t){this._pingPongRenderTarget!==null&&this._dispose(),this._pingPongRenderTarget=pl(e,t,n);const{_lodMax:s}=this;({lodMeshes:this._lodMeshes,sizeLods:this._sizeLods,sigmas:this._sigmas}=np(s)),this._blurMaterial=rp(s,e,t),this._ggxMaterial=ip(s,e,t)}return r}_compileMaterial(e){const t=new _n(new Qt,e);this._renderer.compile(t,ar)}_sceneToCubeUV(e,t,n,r,s){const c=new rn(90,1,t,n),l=[1,-1,1,1,1,1],u=[1,1,1,-1,-1,-1],d=this._renderer,h=d.autoClear,f=d.toneMapping;d.getClearColor(dl),d.toneMapping=An,d.autoClear=!1,d.state.buffers.depth.getReversed()&&(d.setRenderTarget(r),d.clearDepth(),d.setRenderTarget(null)),this._backgroundBox===null&&(this._backgroundBox=new _n(new ii,new oc({name:"PMREM.Background",side:qt,depthWrite:!1,depthTest:!1})));const y=this._backgroundBox,g=y.material;let m=!1;const b=e.background;b?b.isColor&&(g.color.copy(b),e.background=null,m=!0):(g.color.copy(dl),m=!0);for(let w=0;w<6;w++){const A=w%3;A===0?(c.up.set(0,l[w],0),c.position.set(s.x,s.y,s.z),c.lookAt(s.x+u[w],s.y,s.z)):A===1?(c.up.set(0,0,l[w]),c.position.set(s.x,s.y,s.z),c.lookAt(s.x,s.y+u[w],s.z)):(c.up.set(0,l[w],0),c.position.set(s.x,s.y,s.z),c.lookAt(s.x,s.y,s.z+u[w]));const U=this._cubeSize;Ni(r,A*U,w>2?U:0,U,U),d.setRenderTarget(r),m&&d.render(y,c),d.render(e,c)}d.toneMapping=f,d.autoClear=h,e.background=b}_textureToCubeUV(e,t){const n=this._renderer,r=e.mapping===_i||e.mapping===Wi;r?(this._cubemapMaterial===null&&(this._cubemapMaterial=gl()),this._cubemapMaterial.uniforms.flipEnvMap.value=e.isRenderTargetTexture===!1?-1:1):this._equirectMaterial===null&&(this._equirectMaterial=ml());const s=r?this._cubemapMaterial:this._equirectMaterial,a=this._lodMeshes[0];a.material=s;const o=s.uniforms;o.envMap.value=e;const c=this._cubeSize;Ni(t,0,0,3*c,2*c),n.setRenderTarget(t),n.render(a,ar)}_applyPMREM(e){const t=this._renderer,n=t.autoClear;t.autoClear=!1;const r=this._lodMeshes.length;for(let s=1;s<r;s++)this._applyGGXFilter(e,s-1,s);t.autoClear=n}_applyGGXFilter(e,t,n){const r=this._renderer,s=this._pingPongRenderTarget,a=this._ggxMaterial,o=this._lodMeshes[n];o.material=a;const c=a.uniforms,l=n/(this._lodMeshes.length-1),u=t/(this._lodMeshes.length-1),d=Math.sqrt(l*l-u*u),h=0+l*1.25,f=d*h,{_lodMax:_}=this,y=this._sizeLods[n],g=3*y*(n>_-ni?n-_+ni:0),m=4*(this._cubeSize-y);c.envMap.value=e.texture,c.roughness.value=f,c.mipInt.value=_-t,Ni(s,g,m,3*y,2*y),r.setRenderTarget(s),r.render(o,ar),c.envMap.value=s.texture,c.roughness.value=0,c.mipInt.value=_-n,Ni(e,g,m,3*y,2*y),r.setRenderTarget(e),r.render(o,ar)}_blur(e,t,n,r,s){const a=this._pingPongRenderTarget;this._halfBlur(e,a,t,n,r,"latitudinal",s),this._halfBlur(a,e,n,n,r,"longitudinal",s)}_halfBlur(e,t,n,r,s,a,o){const c=this._renderer,l=this._blurMaterial;a!=="latitudinal"&&a!=="longitudinal"&&ot("blur direction must be either latitudinal or longitudinal!");const u=3,d=this._lodMeshes[r];d.material=l;const h=l.uniforms,f=this._sizeLods[n]-1,_=isFinite(s)?Math.PI/(2*f):2*Math.PI/(2*pi-1),y=s/_,g=isFinite(s)?1+Math.floor(u*y):pi;g>pi&&Xe(`sigmaRadians, ${s}, is too large and will clip, as it requested ${g} samples when the maximum is set to ${pi}`);const m=[];let b=0;for(let N=0;N<pi;++N){const S=N/y,T=Math.exp(-S*S/2);m.push(T),N===0?b+=T:N<g&&(b+=2*T)}for(let N=0;N<m.length;N++)m[N]=m[N]/b;h.envMap.value=e.texture,h.samples.value=g,h.weights.value=m,h.latitudinal.value=a==="latitudinal",o&&(h.poleAxis.value=o);const{_lodMax:w}=this;h.dTheta.value=_,h.mipInt.value=w-n;const A=this._sizeLods[r],U=3*A*(r>w-ni?r-w+ni:0),L=4*(this._cubeSize-A);Ni(t,U,L,3*A,2*A),c.setRenderTarget(t),c.render(d,ar)}}function np(i){const e=[],t=[],n=[];let r=i;const s=i-ni+1+ul.length;for(let a=0;a<s;a++){const o=Math.pow(2,r);e.push(o);let c=1/o;a>i-ni?c=ul[a-i+ni-1]:a===0&&(c=0),t.push(c);const l=1/(o-2),u=-l,d=1+l,h=[u,u,d,u,d,d,u,u,d,d,u,d],f=6,_=6,y=3,g=2,m=1,b=new Float32Array(y*_*f),w=new Float32Array(g*_*f),A=new Float32Array(m*_*f);for(let L=0;L<f;L++){const N=L%3*2/3-1,S=L>2?0:-1,T=[N,S,0,N+2/3,S,0,N+2/3,S+1,0,N,S,0,N+2/3,S+1,0,N,S+1,0];b.set(T,y*_*L),w.set(h,g*_*L);const G=[L,L,L,L,L,L];A.set(G,m*_*L)}const U=new Qt;U.setAttribute("position",new on(b,y)),U.setAttribute("uv",new on(w,g)),U.setAttribute("faceIndex",new on(A,m)),n.push(new _n(U,null)),r>ni&&r--}return{lodMeshes:n,sizeLods:e,sigmas:t}}function pl(i,e,t){const n=new wn(i,e,t);return n.texture.mapping=gs,n.texture.name="PMREM.cubeUv",n.scissorTest=!0,n}function Ni(i,e,t,n,r){i.viewport.set(e,t,n,r),i.scissor.set(e,t,n,r)}function ip(i,e,t){return new Pn({name:"PMREMGGXConvolution",defines:{GGX_SAMPLES:ep,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},roughness:{value:0},mipInt:{value:0}},vertexShader:vs(),fragmentShader:`

			precision highp float;
			precision highp int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform float roughness;
			uniform float mipInt;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			#define PI 3.14159265359

			// Van der Corput radical inverse
			float radicalInverse_VdC(uint bits) {
				bits = (bits << 16u) | (bits >> 16u);
				bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
				bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
				bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
				bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
				return float(bits) * 2.3283064365386963e-10; // / 0x100000000
			}

			// Hammersley sequence
			vec2 hammersley(uint i, uint N) {
				return vec2(float(i) / float(N), radicalInverse_VdC(i));
			}

			// GGX VNDF importance sampling (Eric Heitz 2018)
			// "Sampling the GGX Distribution of Visible Normals"
			// https://jcgt.org/published/0007/04/01/
			vec3 importanceSampleGGX_VNDF(vec2 Xi, vec3 V, float roughness) {
				float alpha = roughness * roughness;

				// Section 4.1: Orthonormal basis
				vec3 T1 = vec3(1.0, 0.0, 0.0);
				vec3 T2 = cross(V, T1);

				// Section 4.2: Parameterization of projected area
				float r = sqrt(Xi.x);
				float phi = 2.0 * PI * Xi.y;
				float t1 = r * cos(phi);
				float t2 = r * sin(phi);
				float s = 0.5 * (1.0 + V.z);
				t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

				// Section 4.3: Reprojection onto hemisphere
				vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;

				// Section 3.4: Transform back to ellipsoid configuration
				return normalize(vec3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));
			}

			void main() {
				vec3 N = normalize(vOutputDirection);
				vec3 V = N; // Assume view direction equals normal for pre-filtering

				vec3 prefilteredColor = vec3(0.0);
				float totalWeight = 0.0;

				// For very low roughness, just sample the environment directly
				if (roughness < 0.001) {
					gl_FragColor = vec4(bilinearCubeUV(envMap, N, mipInt), 1.0);
					return;
				}

				// Tangent space basis for VNDF sampling
				vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
				vec3 tangent = normalize(cross(up, N));
				vec3 bitangent = cross(N, tangent);

				for(uint i = 0u; i < uint(GGX_SAMPLES); i++) {
					vec2 Xi = hammersley(i, uint(GGX_SAMPLES));

					// For PMREM, V = N, so in tangent space V is always (0, 0, 1)
					vec3 H_tangent = importanceSampleGGX_VNDF(Xi, vec3(0.0, 0.0, 1.0), roughness);

					// Transform H back to world space
					vec3 H = normalize(tangent * H_tangent.x + bitangent * H_tangent.y + N * H_tangent.z);
					vec3 L = normalize(2.0 * dot(V, H) * H - V);

					float NdotL = max(dot(N, L), 0.0);

					if(NdotL > 0.0) {
						// Sample environment at fixed mip level
						// VNDF importance sampling handles the distribution filtering
						vec3 sampleColor = bilinearCubeUV(envMap, L, mipInt);

						// Weight by NdotL for the split-sum approximation
						// VNDF PDF naturally accounts for the visible microfacet distribution
						prefilteredColor += sampleColor * NdotL;
						totalWeight += NdotL;
					}
				}

				if (totalWeight > 0.0) {
					prefilteredColor = prefilteredColor / totalWeight;
				}

				gl_FragColor = vec4(prefilteredColor, 1.0);
			}
		`,blending:zn,depthTest:!1,depthWrite:!1})}function rp(i,e,t){const n=new Float32Array(pi),r=new q(0,1,0);return new Pn({name:"SphericalGaussianBlur",defines:{n:pi,CUBEUV_TEXEL_WIDTH:1/e,CUBEUV_TEXEL_HEIGHT:1/t,CUBEUV_MAX_MIP:`${i}.0`},uniforms:{envMap:{value:null},samples:{value:1},weights:{value:n},latitudinal:{value:!1},dTheta:{value:0},mipInt:{value:0},poleAxis:{value:r}},vertexShader:vs(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;
			uniform int samples;
			uniform float weights[ n ];
			uniform bool latitudinal;
			uniform float dTheta;
			uniform float mipInt;
			uniform vec3 poleAxis;

			#define ENVMAP_TYPE_CUBE_UV
			#include <cube_uv_reflection_fragment>

			vec3 getSample( float theta, vec3 axis ) {

				float cosTheta = cos( theta );
				// Rodrigues' axis-angle rotation
				vec3 sampleDirection = vOutputDirection * cosTheta
					+ cross( axis, vOutputDirection ) * sin( theta )
					+ axis * dot( axis, vOutputDirection ) * ( 1.0 - cosTheta );

				return bilinearCubeUV( envMap, sampleDirection, mipInt );

			}

			void main() {

				vec3 axis = latitudinal ? poleAxis : cross( poleAxis, vOutputDirection );

				if ( all( equal( axis, vec3( 0.0 ) ) ) ) {

					axis = vec3( vOutputDirection.z, 0.0, - vOutputDirection.x );

				}

				axis = normalize( axis );

				gl_FragColor = vec4( 0.0, 0.0, 0.0, 1.0 );
				gl_FragColor.rgb += weights[ 0 ] * getSample( 0.0, axis );

				for ( int i = 1; i < n; i++ ) {

					if ( i >= samples ) {

						break;

					}

					float theta = dTheta * float( i );
					gl_FragColor.rgb += weights[ i ] * getSample( -1.0 * theta, axis );
					gl_FragColor.rgb += weights[ i ] * getSample( theta, axis );

				}

			}
		`,blending:zn,depthTest:!1,depthWrite:!1})}function ml(){return new Pn({name:"EquirectangularToCubeUV",uniforms:{envMap:{value:null}},vertexShader:vs(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			varying vec3 vOutputDirection;

			uniform sampler2D envMap;

			#include <common>

			void main() {

				vec3 outputDirection = normalize( vOutputDirection );
				vec2 uv = equirectUv( outputDirection );

				gl_FragColor = vec4( texture2D ( envMap, uv ).rgb, 1.0 );

			}
		`,blending:zn,depthTest:!1,depthWrite:!1})}function gl(){return new Pn({name:"CubemapToCubeUV",uniforms:{envMap:{value:null},flipEnvMap:{value:-1}},vertexShader:vs(),fragmentShader:`

			precision mediump float;
			precision mediump int;

			uniform float flipEnvMap;

			varying vec3 vOutputDirection;

			uniform samplerCube envMap;

			void main() {

				gl_FragColor = textureCube( envMap, vec3( flipEnvMap * vOutputDirection.x, vOutputDirection.yz ) );

			}
		`,blending:zn,depthTest:!1,depthWrite:!1})}function vs(){return`

		precision mediump float;
		precision mediump int;

		attribute float faceIndex;

		varying vec3 vOutputDirection;

		// RH coordinate system; PMREM face-indexing convention
		vec3 getDirection( vec2 uv, float face ) {

			uv = 2.0 * uv - 1.0;

			vec3 direction = vec3( uv, 1.0 );

			if ( face == 0.0 ) {

				direction = direction.zyx; // ( 1, v, u ) pos x

			} else if ( face == 1.0 ) {

				direction = direction.xzy;
				direction.xz *= -1.0; // ( -u, 1, -v ) pos y

			} else if ( face == 2.0 ) {

				direction.x *= -1.0; // ( -u, v, 1 ) pos z

			} else if ( face == 3.0 ) {

				direction = direction.zyx;
				direction.xz *= -1.0; // ( -1, v, -u ) neg x

			} else if ( face == 4.0 ) {

				direction = direction.xzy;
				direction.xy *= -1.0; // ( -u, -1, v ) neg y

			} else if ( face == 5.0 ) {

				direction.z *= -1.0; // ( u, v, -1 ) neg z

			}

			return direction;

		}

		void main() {

			vOutputDirection = getDirection( uv, faceIndex );
			gl_Position = vec4( position, 1.0 );

		}
	`}class mc extends wn{constructor(e=1,t={}){super(e,e,t),this.isWebGLCubeRenderTarget=!0;const n={width:e,height:e,depth:1},r=[n,n,n,n,n,n];this.texture=new cc(r),this._setTextureOptions(t),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(e,t){this.texture.type=t.type,this.texture.colorSpace=t.colorSpace,this.texture.generateMipmaps=t.generateMipmaps,this.texture.minFilter=t.minFilter,this.texture.magFilter=t.magFilter;const n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},r=new ii(5,5,5),s=new Pn({name:"CubemapFromEquirect",uniforms:Xi(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:qt,blending:zn});s.uniforms.tEquirect.value=t;const a=new _n(r,s),o=t.minFilter;return t.minFilter===ti&&(t.minFilter=bt),new lu(1,10,this).update(e,a),t.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(e,t=!0,n=!0,r=!0){const s=e.getRenderTarget();for(let a=0;a<6;a++)e.setRenderTarget(this,a),e.clear(t,n,r);e.setRenderTarget(s)}}function sp(i){let e=new WeakMap,t=new WeakMap,n=null;function r(h,f=!1){return h==null?null:f?a(h):s(h)}function s(h){if(h&&h.isTexture){const f=h.mapping;if(f===ss||f===ys)if(e.has(h)){const _=e.get(h).texture;return o(_,h.mapping)}else{const _=h.image;if(_&&_.height>0){const y=new mc(_.height);return y.fromEquirectangularTexture(i,h),e.set(h,y),h.addEventListener("dispose",l),o(y.texture,h.mapping)}else return null}}return h}function a(h){if(h&&h.isTexture){const f=h.mapping,_=f===ss||f===ys,y=f===_i||f===Wi;if(_||y){let g=t.get(h);const m=g!==void 0?g.texture.pmremVersion:0;if(h.isRenderTargetTexture&&h.pmremVersion!==m)return n===null&&(n=new fl(i)),g=_?n.fromEquirectangular(h,g):n.fromCubemap(h,g),g.texture.pmremVersion=h.pmremVersion,t.set(h,g),g.texture;if(g!==void 0)return g.texture;{const b=h.image;return _&&b&&b.height>0||y&&b&&c(b)?(n===null&&(n=new fl(i)),g=_?n.fromEquirectangular(h):n.fromCubemap(h),g.texture.pmremVersion=h.pmremVersion,t.set(h,g),h.addEventListener("dispose",u),g.texture):null}}}return h}function o(h,f){return f===ss?h.mapping=_i:f===ys&&(h.mapping=Wi),h}function c(h){let f=0;const _=6;for(let y=0;y<_;y++)h[y]!==void 0&&f++;return f===_}function l(h){const f=h.target;f.removeEventListener("dispose",l);const _=e.get(f);_!==void 0&&(e.delete(f),_.dispose())}function u(h){const f=h.target;f.removeEventListener("dispose",u);const _=t.get(f);_!==void 0&&(t.delete(f),_.dispose())}function d(){e=new WeakMap,t=new WeakMap,n!==null&&(n.dispose(),n=null)}return{get:r,dispose:d}}function ap(i){const e={};function t(n){if(e[n]!==void 0)return e[n];const r=i.getExtension(n);return e[n]=r,r}return{has:function(n){return t(n)!==null},init:function(){t("EXT_color_buffer_float"),t("WEBGL_clip_cull_distance"),t("OES_texture_float_linear"),t("EXT_color_buffer_half_float"),t("WEBGL_multisampled_render_to_texture"),t("WEBGL_render_shared_exponent")},get:function(n){const r=t(n);return r===null&&fs("WebGLRenderer: "+n+" extension not supported."),r}}}function op(i,e,t,n){const r={},s=new WeakMap;function a(d){const h=d.target;h.index!==null&&e.remove(h.index);for(const _ in h.attributes)e.remove(h.attributes[_]);h.removeEventListener("dispose",a),delete r[h.id];const f=s.get(h);f&&(e.remove(f),s.delete(h)),n.releaseStatesOfGeometry(h),h.isInstancedBufferGeometry===!0&&delete h._maxInstanceCount,t.memory.geometries--}function o(d,h){return r[h.id]===!0||(h.addEventListener("dispose",a),r[h.id]=!0,t.memory.geometries++),h}function c(d){const h=d.attributes;for(const f in h)e.update(h[f],i.ARRAY_BUFFER)}function l(d){const h=[],f=d.index,_=d.attributes.position;let y=0;if(_===void 0)return;if(f!==null){const b=f.array;y=f.version;for(let w=0,A=b.length;w<A;w+=3){const U=b[w+0],L=b[w+1],N=b[w+2];h.push(U,L,L,N,N,U)}}else{const b=_.array;y=_.version;for(let w=0,A=b.length/3-1;w<A;w+=3){const U=w+0,L=w+1,N=w+2;h.push(U,L,L,N,N,U)}}const g=new(_.count>=65535?ac:sc)(h,1);g.version=y;const m=s.get(d);m&&e.remove(m),s.set(d,g)}function u(d){const h=s.get(d);if(h){const f=d.index;f!==null&&h.version<f.version&&l(d)}else l(d);return s.get(d)}return{get:o,update:c,getWireframeAttribute:u}}function lp(i,e,t){let n;function r(h){n=h}let s,a;function o(h){s=h.type,a=h.bytesPerElement}function c(h,f){i.drawElements(n,f,s,h*a),t.update(f,n,1)}function l(h,f,_){_!==0&&(i.drawElementsInstanced(n,f,s,h*a,_),t.update(f,n,_))}function u(h,f,_){if(_===0)return;e.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,f,0,s,h,0,_);let g=0;for(let m=0;m<_;m++)g+=f[m];t.update(g,n,1)}function d(h,f,_,y){if(_===0)return;const g=e.get("WEBGL_multi_draw");if(g===null)for(let m=0;m<h.length;m++)l(h[m]/a,f[m],y[m]);else{g.multiDrawElementsInstancedWEBGL(n,f,0,s,h,0,y,0,_);let m=0;for(let b=0;b<_;b++)m+=f[b]*y[b];t.update(m,n,1)}}this.setMode=r,this.setIndex=o,this.render=c,this.renderInstances=l,this.renderMultiDraw=u,this.renderMultiDrawInstances=d}function cp(i){const e={geometries:0,textures:0},t={frame:0,calls:0,triangles:0,points:0,lines:0};function n(s,a,o){switch(t.calls++,a){case i.TRIANGLES:t.triangles+=o*(s/3);break;case i.LINES:t.lines+=o*(s/2);break;case i.LINE_STRIP:t.lines+=o*(s-1);break;case i.LINE_LOOP:t.lines+=o*s;break;case i.POINTS:t.points+=o*s;break;default:ot("WebGLInfo: Unknown draw mode:",a);break}}function r(){t.calls=0,t.triangles=0,t.points=0,t.lines=0}return{memory:e,render:t,programs:null,autoReset:!0,reset:r,update:n}}function hp(i,e,t){const n=new WeakMap,r=new Et;function s(a,o,c){const l=a.morphTargetInfluences,u=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,d=u!==void 0?u.length:0;let h=n.get(o);if(h===void 0||h.count!==d){let G=function(){S.dispose(),n.delete(o),o.removeEventListener("dispose",G)};var f=G;h!==void 0&&h.texture.dispose();const _=o.morphAttributes.position!==void 0,y=o.morphAttributes.normal!==void 0,g=o.morphAttributes.color!==void 0,m=o.morphAttributes.position||[],b=o.morphAttributes.normal||[],w=o.morphAttributes.color||[];let A=0;_===!0&&(A=1),y===!0&&(A=2),g===!0&&(A=3);let U=o.attributes.position.count*A,L=1;U>e.maxTextureSize&&(L=Math.ceil(U/e.maxTextureSize),U=e.maxTextureSize);const N=new Float32Array(U*L*4*d),S=new ic(N,U,L,d);S.type=Yt,S.needsUpdate=!0;const T=A*4;for(let D=0;D<d;D++){const O=m[D],V=b[D],K=w[D],Y=U*L*4*D;for(let Z=0;Z<O.count;Z++){const X=Z*T;_===!0&&(r.fromBufferAttribute(O,Z),N[Y+X+0]=r.x,N[Y+X+1]=r.y,N[Y+X+2]=r.z,N[Y+X+3]=0),y===!0&&(r.fromBufferAttribute(V,Z),N[Y+X+4]=r.x,N[Y+X+5]=r.y,N[Y+X+6]=r.z,N[Y+X+7]=0),g===!0&&(r.fromBufferAttribute(K,Z),N[Y+X+8]=r.x,N[Y+X+9]=r.y,N[Y+X+10]=r.z,N[Y+X+11]=K.itemSize===4?r.w:1)}}h={count:d,texture:S,size:new $e(U,L)},n.set(o,h),o.addEventListener("dispose",G)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)c.getUniforms().setValue(i,"morphTexture",a.morphTexture,t);else{let _=0;for(let g=0;g<l.length;g++)_+=l[g];const y=o.morphTargetsRelative?1:1-_;c.getUniforms().setValue(i,"morphTargetBaseInfluence",y),c.getUniforms().setValue(i,"morphTargetInfluences",l)}c.getUniforms().setValue(i,"morphTargetsTexture",h.texture,t),c.getUniforms().setValue(i,"morphTargetsTextureSize",h.size)}return{update:s}}function up(i,e,t,n,r){let s=new WeakMap;function a(l){const u=r.render.frame,d=l.geometry,h=e.get(l,d);if(s.get(h)!==u&&(e.update(h),s.set(h,u)),l.isInstancedMesh&&(l.hasEventListener("dispose",c)===!1&&l.addEventListener("dispose",c),s.get(l)!==u&&(t.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&t.update(l.instanceColor,i.ARRAY_BUFFER),s.set(l,u))),l.isSkinnedMesh){const f=l.skeleton;s.get(f)!==u&&(f.update(),s.set(f,u))}return h}function o(){s=new WeakMap}function c(l){const u=l.target;u.removeEventListener("dispose",c),n.releaseStatesOfObject(u),t.remove(u.instanceMatrix),u.instanceColor!==null&&t.remove(u.instanceColor)}return{update:a,dispose:o}}const dp={[Gl]:"LINEAR_TONE_MAPPING",[Hl]:"REINHARD_TONE_MAPPING",[Vl]:"CINEON_TONE_MAPPING",[Wl]:"ACES_FILMIC_TONE_MAPPING",[Yl]:"AGX_TONE_MAPPING",[ql]:"NEUTRAL_TONE_MAPPING",[Xl]:"CUSTOM_TONE_MAPPING"};function fp(i,e,t,n,r){const s=new wn(e,t,{type:i,depthBuffer:n,stencilBuffer:r}),a=new wn(e,t,{type:Jt,depthBuffer:!1,stencilBuffer:!1}),o=new Qt;o.setAttribute("position",new It([-1,3,0,-1,-1,0,3,-1,0],3)),o.setAttribute("uv",new It([0,2,0,0,2,0],2));const c=new $h({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),l=new _n(o,c),u=new Mo(-1,1,1,-1,0,1);let d=null,h=null,f=!1,_,y=null,g=[],m=!1;this.setSize=function(b,w){s.setSize(b,w),a.setSize(b,w);for(let A=0;A<g.length;A++){const U=g[A];U.setSize&&U.setSize(b,w)}},this.setEffects=function(b){g=b,m=g.length>0&&g[0].isRenderPass===!0;const w=s.width,A=s.height;for(let U=0;U<g.length;U++){const L=g[U];L.setSize&&L.setSize(w,A)}},this.begin=function(b,w){if(f||b.toneMapping===An&&g.length===0)return!1;if(y=w,w!==null){const A=w.width,U=w.height;(s.width!==A||s.height!==U)&&this.setSize(A,U)}return m===!1&&b.setRenderTarget(s),_=b.toneMapping,b.toneMapping=An,!0},this.hasRenderPass=function(){return m},this.end=function(b,w){b.toneMapping=_,f=!0;let A=s,U=a;for(let L=0;L<g.length;L++){const N=g[L];if(N.enabled!==!1&&(N.render(b,U,A,w),N.needsSwap!==!1)){const S=A;A=U,U=S}}if(d!==b.outputColorSpace||h!==b.toneMapping){d=b.outputColorSpace,h=b.toneMapping,c.defines={},lt.getTransfer(d)===ft&&(c.defines.SRGB_TRANSFER="");const L=dp[h];L&&(c.defines[L]=""),c.needsUpdate=!0}c.uniforms.tDiffuse.value=A.texture,b.setRenderTarget(y),b.render(l,u),y=null,f=!1},this.isCompositing=function(){return f},this.dispose=function(){s.dispose(),a.dispose(),o.dispose(),c.dispose()}}const gc=new Ht,Ja=new xr(1,1),_c=new ic,xc=new vh,vc=new cc,_l=[],xl=[],vl=new Float32Array(16),Sl=new Float32Array(9),Ml=new Float32Array(4);function Zi(i,e,t){const n=i[0];if(n<=0||n>0)return i;const r=e*t;let s=_l[r];if(s===void 0&&(s=new Float32Array(r),_l[r]=s),e!==0){n.toArray(s,0);for(let a=1,o=0;a!==e;++a)o+=t,i[a].toArray(s,o)}return s}function Ct(i,e){if(i.length!==e.length)return!1;for(let t=0,n=i.length;t<n;t++)if(i[t]!==e[t])return!1;return!0}function Rt(i,e){for(let t=0,n=e.length;t<n;t++)i[t]=e[t]}function Ss(i,e){let t=xl[e];t===void 0&&(t=new Int32Array(e),xl[e]=t);for(let n=0;n!==e;++n)t[n]=i.allocateTextureUnit();return t}function pp(i,e){const t=this.cache;t[0]!==e&&(i.uniform1f(this.addr,e),t[0]=e)}function mp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2f(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Ct(t,e))return;i.uniform2fv(this.addr,e),Rt(t,e)}}function gp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3f(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else if(e.r!==void 0)(t[0]!==e.r||t[1]!==e.g||t[2]!==e.b)&&(i.uniform3f(this.addr,e.r,e.g,e.b),t[0]=e.r,t[1]=e.g,t[2]=e.b);else{if(Ct(t,e))return;i.uniform3fv(this.addr,e),Rt(t,e)}}function _p(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4f(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Ct(t,e))return;i.uniform4fv(this.addr,e),Rt(t,e)}}function xp(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Ct(t,e))return;i.uniformMatrix2fv(this.addr,!1,e),Rt(t,e)}else{if(Ct(t,n))return;Ml.set(n),i.uniformMatrix2fv(this.addr,!1,Ml),Rt(t,n)}}function vp(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Ct(t,e))return;i.uniformMatrix3fv(this.addr,!1,e),Rt(t,e)}else{if(Ct(t,n))return;Sl.set(n),i.uniformMatrix3fv(this.addr,!1,Sl),Rt(t,n)}}function Sp(i,e){const t=this.cache,n=e.elements;if(n===void 0){if(Ct(t,e))return;i.uniformMatrix4fv(this.addr,!1,e),Rt(t,e)}else{if(Ct(t,n))return;vl.set(n),i.uniformMatrix4fv(this.addr,!1,vl),Rt(t,n)}}function Mp(i,e){const t=this.cache;t[0]!==e&&(i.uniform1i(this.addr,e),t[0]=e)}function yp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2i(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Ct(t,e))return;i.uniform2iv(this.addr,e),Rt(t,e)}}function Ep(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3i(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Ct(t,e))return;i.uniform3iv(this.addr,e),Rt(t,e)}}function bp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4i(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Ct(t,e))return;i.uniform4iv(this.addr,e),Rt(t,e)}}function Tp(i,e){const t=this.cache;t[0]!==e&&(i.uniform1ui(this.addr,e),t[0]=e)}function Ap(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y)&&(i.uniform2ui(this.addr,e.x,e.y),t[0]=e.x,t[1]=e.y);else{if(Ct(t,e))return;i.uniform2uiv(this.addr,e),Rt(t,e)}}function wp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z)&&(i.uniform3ui(this.addr,e.x,e.y,e.z),t[0]=e.x,t[1]=e.y,t[2]=e.z);else{if(Ct(t,e))return;i.uniform3uiv(this.addr,e),Rt(t,e)}}function Cp(i,e){const t=this.cache;if(e.x!==void 0)(t[0]!==e.x||t[1]!==e.y||t[2]!==e.z||t[3]!==e.w)&&(i.uniform4ui(this.addr,e.x,e.y,e.z,e.w),t[0]=e.x,t[1]=e.y,t[2]=e.z,t[3]=e.w);else{if(Ct(t,e))return;i.uniform4uiv(this.addr,e),Rt(t,e)}}function Rp(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r);let s;this.type===i.SAMPLER_2D_SHADOW?(Ja.compareFunction=t.isReversedDepthBuffer()?fo:uo,s=Ja):s=gc,t.setTexture2D(e||s,r)}function Pp(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r),t.setTexture3D(e||xc,r)}function Dp(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r),t.setTextureCube(e||vc,r)}function Ip(i,e,t){const n=this.cache,r=t.allocateTextureUnit();n[0]!==r&&(i.uniform1i(this.addr,r),n[0]=r),t.setTexture2DArray(e||_c,r)}function Lp(i){switch(i){case 5126:return pp;case 35664:return mp;case 35665:return gp;case 35666:return _p;case 35674:return xp;case 35675:return vp;case 35676:return Sp;case 5124:case 35670:return Mp;case 35667:case 35671:return yp;case 35668:case 35672:return Ep;case 35669:case 35673:return bp;case 5125:return Tp;case 36294:return Ap;case 36295:return wp;case 36296:return Cp;case 35678:case 36198:case 36298:case 36306:case 35682:return Rp;case 35679:case 36299:case 36307:return Pp;case 35680:case 36300:case 36308:case 36293:return Dp;case 36289:case 36303:case 36311:case 36292:return Ip}}function Up(i,e){i.uniform1fv(this.addr,e)}function Fp(i,e){const t=Zi(e,this.size,2);i.uniform2fv(this.addr,t)}function Np(i,e){const t=Zi(e,this.size,3);i.uniform3fv(this.addr,t)}function Op(i,e){const t=Zi(e,this.size,4);i.uniform4fv(this.addr,t)}function Bp(i,e){const t=Zi(e,this.size,4);i.uniformMatrix2fv(this.addr,!1,t)}function kp(i,e){const t=Zi(e,this.size,9);i.uniformMatrix3fv(this.addr,!1,t)}function zp(i,e){const t=Zi(e,this.size,16);i.uniformMatrix4fv(this.addr,!1,t)}function Gp(i,e){i.uniform1iv(this.addr,e)}function Hp(i,e){i.uniform2iv(this.addr,e)}function Vp(i,e){i.uniform3iv(this.addr,e)}function Wp(i,e){i.uniform4iv(this.addr,e)}function Xp(i,e){i.uniform1uiv(this.addr,e)}function Yp(i,e){i.uniform2uiv(this.addr,e)}function qp(i,e){i.uniform3uiv(this.addr,e)}function Zp(i,e){i.uniform4uiv(this.addr,e)}function $p(i,e,t){const n=this.cache,r=e.length,s=Ss(t,r);Ct(n,s)||(i.uniform1iv(this.addr,s),Rt(n,s));let a;this.type===i.SAMPLER_2D_SHADOW?a=Ja:a=gc;for(let o=0;o!==r;++o)t.setTexture2D(e[o]||a,s[o])}function jp(i,e,t){const n=this.cache,r=e.length,s=Ss(t,r);Ct(n,s)||(i.uniform1iv(this.addr,s),Rt(n,s));for(let a=0;a!==r;++a)t.setTexture3D(e[a]||xc,s[a])}function Kp(i,e,t){const n=this.cache,r=e.length,s=Ss(t,r);Ct(n,s)||(i.uniform1iv(this.addr,s),Rt(n,s));for(let a=0;a!==r;++a)t.setTextureCube(e[a]||vc,s[a])}function Jp(i,e,t){const n=this.cache,r=e.length,s=Ss(t,r);Ct(n,s)||(i.uniform1iv(this.addr,s),Rt(n,s));for(let a=0;a!==r;++a)t.setTexture2DArray(e[a]||_c,s[a])}function Qp(i){switch(i){case 5126:return Up;case 35664:return Fp;case 35665:return Np;case 35666:return Op;case 35674:return Bp;case 35675:return kp;case 35676:return zp;case 5124:case 35670:return Gp;case 35667:case 35671:return Hp;case 35668:case 35672:return Vp;case 35669:case 35673:return Wp;case 5125:return Xp;case 36294:return Yp;case 36295:return qp;case 36296:return Zp;case 35678:case 36198:case 36298:case 36306:case 35682:return $p;case 35679:case 36299:case 36307:return jp;case 35680:case 36300:case 36308:case 36293:return Kp;case 36289:case 36303:case 36311:case 36292:return Jp}}class em{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.setValue=Lp(t.type)}}class tm{constructor(e,t,n){this.id=e,this.addr=n,this.cache=[],this.type=t.type,this.size=t.size,this.setValue=Qp(t.type)}}class nm{constructor(e){this.id=e,this.seq=[],this.map={}}setValue(e,t,n){const r=this.seq;for(let s=0,a=r.length;s!==a;++s){const o=r[s];o.setValue(e,t[o.id],n)}}}const ea=/(\w+)(\])?(\[|\.)?/g;function yl(i,e){i.seq.push(e),i.map[e.id]=e}function im(i,e,t){const n=i.name,r=n.length;for(ea.lastIndex=0;;){const s=ea.exec(n),a=ea.lastIndex;let o=s[1];const c=s[2]==="]",l=s[3];if(c&&(o=o|0),l===void 0||l==="["&&a+2===r){yl(t,l===void 0?new em(o,i,e):new tm(o,i,e));break}else{let d=t.map[o];d===void 0&&(d=new nm(o),yl(t,d)),t=d}}}class hs{constructor(e,t){this.seq=[],this.map={};const n=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let a=0;a<n;++a){const o=e.getActiveUniform(t,a),c=e.getUniformLocation(t,o.name);im(o,c,this)}const r=[],s=[];for(const a of this.seq)a.type===e.SAMPLER_2D_SHADOW||a.type===e.SAMPLER_CUBE_SHADOW||a.type===e.SAMPLER_2D_ARRAY_SHADOW?r.push(a):s.push(a);r.length>0&&(this.seq=r.concat(s))}setValue(e,t,n,r){const s=this.map[t];s!==void 0&&s.setValue(e,n,r)}setOptional(e,t,n){const r=t[n];r!==void 0&&this.setValue(e,n,r)}static upload(e,t,n,r){for(let s=0,a=t.length;s!==a;++s){const o=t[s],c=n[o.id];c.needsUpdate!==!1&&o.setValue(e,c.value,r)}}static seqWithValue(e,t){const n=[];for(let r=0,s=e.length;r!==s;++r){const a=e[r];a.id in t&&n.push(a)}return n}}function El(i,e,t){const n=i.createShader(e);return i.shaderSource(n,t),i.compileShader(n),n}const rm=37297;let sm=0;function am(i,e){const t=i.split(`
`),n=[],r=Math.max(e-6,0),s=Math.min(e+6,t.length);for(let a=r;a<s;a++){const o=a+1;n.push(`${o===e?">":" "} ${o}: ${t[a]}`)}return n.join(`
`)}const bl=new Je;function om(i){lt._getMatrix(bl,lt.workingColorSpace,i);const e=`mat3( ${bl.elements.map(t=>t.toFixed(4))} )`;switch(lt.getTransfer(i)){case us:return[e,"LinearTransferOETF"];case ft:return[e,"sRGBTransferOETF"];default:return Xe("WebGLProgram: Unsupported color space: ",i),[e,"LinearTransferOETF"]}}function Tl(i,e,t){const n=i.getShaderParameter(e,i.COMPILE_STATUS),s=(i.getShaderInfoLog(e)||"").trim();if(n&&s==="")return"";const a=/ERROR: 0:(\d+)/.exec(s);if(a){const o=parseInt(a[1]);return t.toUpperCase()+`

`+s+`

`+am(i.getShaderSource(e),o)}else return s}function lm(i,e){const t=om(e);return[`vec4 ${i}( vec4 value ) {`,`	return ${t[1]}( vec4( value.rgb * ${t[0]}, value.a ) );`,"}"].join(`
`)}const cm={[Gl]:"Linear",[Hl]:"Reinhard",[Vl]:"Cineon",[Wl]:"ACESFilmic",[Yl]:"AgX",[ql]:"Neutral",[Xl]:"Custom"};function hm(i,e){const t=cm[e];return t===void 0?(Xe("WebGLProgram: Unsupported toneMapping:",e),"vec3 "+i+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+i+"( vec3 color ) { return "+t+"ToneMapping( color ); }"}const ts=new q;function um(){lt.getLuminanceCoefficients(ts);const i=ts.x.toFixed(4),e=ts.y.toFixed(4),t=ts.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${e}, ${t} );`,"	return dot( weights, rgb );","}"].join(`
`)}function dm(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(dr).join(`
`)}function fm(i){const e=[];for(const t in i){const n=i[t];n!==!1&&e.push("#define "+t+" "+n)}return e.join(`
`)}function pm(i,e){const t={},n=i.getProgramParameter(e,i.ACTIVE_ATTRIBUTES);for(let r=0;r<n;r++){const s=i.getActiveAttrib(e,r),a=s.name;let o=1;s.type===i.FLOAT_MAT2&&(o=2),s.type===i.FLOAT_MAT3&&(o=3),s.type===i.FLOAT_MAT4&&(o=4),t[a]={type:s.type,location:i.getAttribLocation(e,a),locationSize:o}}return t}function dr(i){return i!==""}function Al(i,e){const t=e.numSpotLightShadows+e.numSpotLightMaps-e.numSpotLightShadowsWithMaps;return i.replace(/NUM_DIR_LIGHTS/g,e.numDirLights).replace(/NUM_SPOT_LIGHTS/g,e.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,e.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,t).replace(/NUM_RECT_AREA_LIGHTS/g,e.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,e.numPointLights).replace(/NUM_HEMI_LIGHTS/g,e.numHemiLights).replace(/NUM_DIR_LIGHT_SHADOWS/g,e.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,e.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,e.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,e.numPointLightShadows)}function wl(i,e){return i.replace(/NUM_CLIPPING_PLANES/g,e.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,e.numClippingPlanes-e.numClipIntersection)}const mm=/^[ \t]*#include +<([\w\d./]+)>/gm;function Qa(i){return i.replace(mm,_m)}const gm=new Map;function _m(i,e){let t=Qe[e];if(t===void 0){const n=gm.get(e);if(n!==void 0)t=Qe[n],Xe('WebGLRenderer: Shader chunk "%s" has been deprecated. Use "%s" instead.',e,n);else throw new Error("Can not resolve #include <"+e+">")}return Qa(t)}const xm=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function Cl(i){return i.replace(xm,vm)}function vm(i,e,t,n){let r="";for(let s=parseInt(e);s<parseInt(t);s++)r+=n.replace(/\[\s*i\s*\]/g,"[ "+s+" ]").replace(/UNROLLED_LOOP_INDEX/g,s);return r}function Rl(i){let e=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?e+=`
#define HIGH_PRECISION`:i.precision==="mediump"?e+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(e+=`
#define LOW_PRECISION`),e}const Sm={[rs]:"SHADOWMAP_TYPE_PCF",[ur]:"SHADOWMAP_TYPE_VSM"};function Mm(i){return Sm[i.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}const ym={[_i]:"ENVMAP_TYPE_CUBE",[Wi]:"ENVMAP_TYPE_CUBE",[gs]:"ENVMAP_TYPE_CUBE_UV"};function Em(i){return i.envMap===!1?"ENVMAP_TYPE_CUBE":ym[i.envMapMode]||"ENVMAP_TYPE_CUBE"}const bm={[Wi]:"ENVMAP_MODE_REFRACTION"};function Tm(i){return i.envMap===!1?"ENVMAP_MODE_REFLECTION":bm[i.envMapMode]||"ENVMAP_MODE_REFLECTION"}const Am={[ro]:"ENVMAP_BLENDING_MULTIPLY",[jc]:"ENVMAP_BLENDING_MIX",[Kc]:"ENVMAP_BLENDING_ADD"};function wm(i){return i.envMap===!1?"ENVMAP_BLENDING_NONE":Am[i.combine]||"ENVMAP_BLENDING_NONE"}function Cm(i){const e=i.envMapCubeUVHeight;if(e===null)return null;const t=Math.log2(e)-2,n=1/e;return{texelWidth:1/(3*Math.max(Math.pow(2,t),112)),texelHeight:n,maxMip:t}}function Rm(i,e,t,n){const r=i.getContext(),s=t.defines;let a=t.vertexShader,o=t.fragmentShader;const c=Mm(t),l=Em(t),u=Tm(t),d=wm(t),h=Cm(t),f=dm(t),_=fm(s),y=r.createProgram();let g,m,b=t.glslVersion?"#version "+t.glslVersion+`
`:"";t.isRawShaderMaterial?(g=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,_].filter(dr).join(`
`),g.length>0&&(g+=`
`),m=["#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,_].filter(dr).join(`
`),m.length>0&&(m+=`
`)):(g=[Rl(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,_,t.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",t.batching?"#define USE_BATCHING":"",t.batchingColor?"#define USE_BATCHING_COLOR":"",t.instancing?"#define USE_INSTANCING":"",t.instancingColor?"#define USE_INSTANCING_COLOR":"",t.instancingMorph?"#define USE_INSTANCING_MORPH":"",t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.map?"#define USE_MAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+u:"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.displacementMap?"#define USE_DISPLACEMENTMAP":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.mapUv?"#define MAP_UV "+t.mapUv:"",t.alphaMapUv?"#define ALPHAMAP_UV "+t.alphaMapUv:"",t.lightMapUv?"#define LIGHTMAP_UV "+t.lightMapUv:"",t.aoMapUv?"#define AOMAP_UV "+t.aoMapUv:"",t.emissiveMapUv?"#define EMISSIVEMAP_UV "+t.emissiveMapUv:"",t.bumpMapUv?"#define BUMPMAP_UV "+t.bumpMapUv:"",t.normalMapUv?"#define NORMALMAP_UV "+t.normalMapUv:"",t.displacementMapUv?"#define DISPLACEMENTMAP_UV "+t.displacementMapUv:"",t.metalnessMapUv?"#define METALNESSMAP_UV "+t.metalnessMapUv:"",t.roughnessMapUv?"#define ROUGHNESSMAP_UV "+t.roughnessMapUv:"",t.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+t.anisotropyMapUv:"",t.clearcoatMapUv?"#define CLEARCOATMAP_UV "+t.clearcoatMapUv:"",t.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+t.clearcoatNormalMapUv:"",t.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+t.clearcoatRoughnessMapUv:"",t.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+t.iridescenceMapUv:"",t.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+t.iridescenceThicknessMapUv:"",t.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+t.sheenColorMapUv:"",t.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+t.sheenRoughnessMapUv:"",t.specularMapUv?"#define SPECULARMAP_UV "+t.specularMapUv:"",t.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+t.specularColorMapUv:"",t.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+t.specularIntensityMapUv:"",t.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+t.transmissionMapUv:"",t.thicknessMapUv?"#define THICKNESSMAP_UV "+t.thicknessMapUv:"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors?"#define USE_COLOR":"",t.vertexAlphas?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.flatShading?"#define FLAT_SHADED":"",t.skinning?"#define USE_SKINNING":"",t.morphTargets?"#define USE_MORPHTARGETS":"",t.morphNormals&&t.flatShading===!1?"#define USE_MORPHNORMALS":"",t.morphColors?"#define USE_MORPHCOLORS":"",t.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+t.morphTextureStride:"",t.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+t.morphTargetsCount:"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.sizeAttenuation?"#define USE_SIZEATTENUATION":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(dr).join(`
`),m=[Rl(t),"#define SHADER_TYPE "+t.shaderType,"#define SHADER_NAME "+t.shaderName,_,t.useFog&&t.fog?"#define USE_FOG":"",t.useFog&&t.fogExp2?"#define FOG_EXP2":"",t.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",t.map?"#define USE_MAP":"",t.matcap?"#define USE_MATCAP":"",t.envMap?"#define USE_ENVMAP":"",t.envMap?"#define "+l:"",t.envMap?"#define "+u:"",t.envMap?"#define "+d:"",h?"#define CUBEUV_TEXEL_WIDTH "+h.texelWidth:"",h?"#define CUBEUV_TEXEL_HEIGHT "+h.texelHeight:"",h?"#define CUBEUV_MAX_MIP "+h.maxMip+".0":"",t.lightMap?"#define USE_LIGHTMAP":"",t.aoMap?"#define USE_AOMAP":"",t.bumpMap?"#define USE_BUMPMAP":"",t.normalMap?"#define USE_NORMALMAP":"",t.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",t.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",t.emissiveMap?"#define USE_EMISSIVEMAP":"",t.anisotropy?"#define USE_ANISOTROPY":"",t.anisotropyMap?"#define USE_ANISOTROPYMAP":"",t.clearcoat?"#define USE_CLEARCOAT":"",t.clearcoatMap?"#define USE_CLEARCOATMAP":"",t.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",t.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",t.dispersion?"#define USE_DISPERSION":"",t.iridescence?"#define USE_IRIDESCENCE":"",t.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",t.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",t.specularMap?"#define USE_SPECULARMAP":"",t.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",t.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",t.roughnessMap?"#define USE_ROUGHNESSMAP":"",t.metalnessMap?"#define USE_METALNESSMAP":"",t.alphaMap?"#define USE_ALPHAMAP":"",t.alphaTest?"#define USE_ALPHATEST":"",t.alphaHash?"#define USE_ALPHAHASH":"",t.sheen?"#define USE_SHEEN":"",t.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",t.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",t.transmission?"#define USE_TRANSMISSION":"",t.transmissionMap?"#define USE_TRANSMISSIONMAP":"",t.thicknessMap?"#define USE_THICKNESSMAP":"",t.vertexTangents&&t.flatShading===!1?"#define USE_TANGENT":"",t.vertexColors||t.instancingColor?"#define USE_COLOR":"",t.vertexAlphas||t.batchingColor?"#define USE_COLOR_ALPHA":"",t.vertexUv1s?"#define USE_UV1":"",t.vertexUv2s?"#define USE_UV2":"",t.vertexUv3s?"#define USE_UV3":"",t.pointsUvs?"#define USE_POINTS_UV":"",t.gradientMap?"#define USE_GRADIENTMAP":"",t.flatShading?"#define FLAT_SHADED":"",t.doubleSided?"#define DOUBLE_SIDED":"",t.flipSided?"#define FLIP_SIDED":"",t.shadowMapEnabled?"#define USE_SHADOWMAP":"",t.shadowMapEnabled?"#define "+c:"",t.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",t.numLightProbes>0?"#define USE_LIGHT_PROBES":"",t.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",t.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",t.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",t.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",t.toneMapping!==An?"#define TONE_MAPPING":"",t.toneMapping!==An?Qe.tonemapping_pars_fragment:"",t.toneMapping!==An?hm("toneMapping",t.toneMapping):"",t.dithering?"#define DITHERING":"",t.opaque?"#define OPAQUE":"",Qe.colorspace_pars_fragment,lm("linearToOutputTexel",t.outputColorSpace),um(),t.useDepthPacking?"#define DEPTH_PACKING "+t.depthPacking:"",`
`].filter(dr).join(`
`)),a=Qa(a),a=Al(a,t),a=wl(a,t),o=Qa(o),o=Al(o,t),o=wl(o,t),a=Cl(a),o=Cl(o),t.isRawShaderMaterial!==!0&&(b=`#version 300 es
`,g=[f,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+g,m=["#define varying in",t.glslVersion===Po?"":"layout(location = 0) out highp vec4 pc_fragColor;",t.glslVersion===Po?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+m);const w=b+g+a,A=b+m+o,U=El(r,r.VERTEX_SHADER,w),L=El(r,r.FRAGMENT_SHADER,A);r.attachShader(y,U),r.attachShader(y,L),t.index0AttributeName!==void 0?r.bindAttribLocation(y,0,t.index0AttributeName):t.morphTargets===!0&&r.bindAttribLocation(y,0,"position"),r.linkProgram(y);function N(D){if(i.debug.checkShaderErrors){const O=r.getProgramInfoLog(y)||"",V=r.getShaderInfoLog(U)||"",K=r.getShaderInfoLog(L)||"",Y=O.trim(),Z=V.trim(),X=K.trim();let fe=!0,oe=!0;if(r.getProgramParameter(y,r.LINK_STATUS)===!1)if(fe=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(r,y,U,L);else{const ye=Tl(r,U,"vertex"),Ae=Tl(r,L,"fragment");ot("THREE.WebGLProgram: Shader Error "+r.getError()+" - VALIDATE_STATUS "+r.getProgramParameter(y,r.VALIDATE_STATUS)+`

Material Name: `+D.name+`
Material Type: `+D.type+`

Program Info Log: `+Y+`
`+ye+`
`+Ae)}else Y!==""?Xe("WebGLProgram: Program Info Log:",Y):(Z===""||X==="")&&(oe=!1);oe&&(D.diagnostics={runnable:fe,programLog:Y,vertexShader:{log:Z,prefix:g},fragmentShader:{log:X,prefix:m}})}r.deleteShader(U),r.deleteShader(L),S=new hs(r,y),T=pm(r,y)}let S;this.getUniforms=function(){return S===void 0&&N(this),S};let T;this.getAttributes=function(){return T===void 0&&N(this),T};let G=t.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return G===!1&&(G=r.getProgramParameter(y,rm)),G},this.destroy=function(){n.releaseStatesOfProgram(this),r.deleteProgram(y),this.program=void 0},this.type=t.shaderType,this.name=t.shaderName,this.id=sm++,this.cacheKey=e,this.usedTimes=1,this.program=y,this.vertexShader=U,this.fragmentShader=L,this}let Pm=0;class Dm{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(e){const t=e.vertexShader,n=e.fragmentShader,r=this._getShaderStage(t),s=this._getShaderStage(n),a=this._getShaderCacheForMaterial(e);return a.has(r)===!1&&(a.add(r),r.usedTimes++),a.has(s)===!1&&(a.add(s),s.usedTimes++),this}remove(e){const t=this.materialCache.get(e);for(const n of t)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(e),this}getVertexShaderID(e){return this._getShaderStage(e.vertexShader).id}getFragmentShaderID(e){return this._getShaderStage(e.fragmentShader).id}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(e){const t=this.materialCache;let n=t.get(e);return n===void 0&&(n=new Set,t.set(e,n)),n}_getShaderStage(e){const t=this.shaderCache;let n=t.get(e);return n===void 0&&(n=new Im(e),t.set(e,n)),n}}class Im{constructor(e){this.id=Pm++,this.code=e,this.usedTimes=0}}function Lm(i,e,t,n,r,s){const a=new mo,o=new Dm,c=new Set,l=[],u=new Map,d=n.logarithmicDepthBuffer;let h=n.precision;const f={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function _(S){return c.add(S),S===0?"uv":`uv${S}`}function y(S,T,G,D,O){const V=D.fog,K=O.geometry,Y=S.isMeshStandardMaterial||S.isMeshLambertMaterial||S.isMeshPhongMaterial?D.environment:null,Z=S.isMeshStandardMaterial||S.isMeshLambertMaterial&&!S.envMap||S.isMeshPhongMaterial&&!S.envMap,X=e.get(S.envMap||Y,Z),fe=X&&X.mapping===gs?X.image.height:null,oe=f[S.type];S.precision!==null&&(h=n.getMaxPrecision(S.precision),h!==S.precision&&Xe("WebGLProgram.getParameters:",S.precision,"not supported, using",h,"instead."));const ye=K.morphAttributes.position||K.morphAttributes.normal||K.morphAttributes.color,Ae=ye!==void 0?ye.length:0;let ve=0;K.morphAttributes.position!==void 0&&(ve=1),K.morphAttributes.normal!==void 0&&(ve=2),K.morphAttributes.color!==void 0&&(ve=3);let Ge,st,_e,$;if(oe){const ut=bn[oe];Ge=ut.vertexShader,st=ut.fragmentShader}else Ge=S.vertexShader,st=S.fragmentShader,o.update(S),_e=o.getVertexShaderID(S),$=o.getFragmentShaderID(S);const ue=i.getRenderTarget(),de=i.state.buffers.depth.getReversed(),ze=O.isInstancedMesh===!0,Le=O.isBatchedMesh===!0,Fe=!!S.map,xt=!!S.matcap,et=!!X,ct=!!S.aoMap,dt=!!S.lightMap,Ke=!!S.bumpMap,St=!!S.normalMap,B=!!S.displacementMap,Mt=!!S.emissiveMap,at=!!S.metalnessMap,pt=!!S.roughnessMap,De=S.anisotropy>0,C=S.clearcoat>0,v=S.dispersion>0,z=S.iridescence>0,re=S.sheen>0,le=S.transmission>0,ne=De&&!!S.anisotropyMap,Ce=C&&!!S.clearcoatMap,xe=C&&!!S.clearcoatNormalMap,ke=C&&!!S.clearcoatRoughnessMap,Ve=z&&!!S.iridescenceMap,me=z&&!!S.iridescenceThicknessMap,pe=re&&!!S.sheenColorMap,Te=re&&!!S.sheenRoughnessMap,Re=!!S.specularMap,Ee=!!S.specularColorMap,He=!!S.specularIntensityMap,k=le&&!!S.transmissionMap,Se=le&&!!S.thicknessMap,J=!!S.gradientMap,we=!!S.alphaMap,ge=S.alphaTest>0,te=!!S.alphaHash,Pe=!!S.extensions;let qe=An;S.toneMapped&&(ue===null||ue.isXRRenderTarget===!0)&&(qe=i.toneMapping);const mt={shaderID:oe,shaderType:S.type,shaderName:S.name,vertexShader:Ge,fragmentShader:st,defines:S.defines,customVertexShaderID:_e,customFragmentShaderID:$,isRawShaderMaterial:S.isRawShaderMaterial===!0,glslVersion:S.glslVersion,precision:h,batching:Le,batchingColor:Le&&O._colorsTexture!==null,instancing:ze,instancingColor:ze&&O.instanceColor!==null,instancingMorph:ze&&O.morphTexture!==null,outputColorSpace:ue===null?i.outputColorSpace:ue.isXRRenderTarget===!0?ue.texture.colorSpace:jt,alphaToCoverage:!!S.alphaToCoverage,map:Fe,matcap:xt,envMap:et,envMapMode:et&&X.mapping,envMapCubeUVHeight:fe,aoMap:ct,lightMap:dt,bumpMap:Ke,normalMap:St,displacementMap:B,emissiveMap:Mt,normalMapObjectSpace:St&&S.normalMapType===eh,normalMapTangentSpace:St&&S.normalMapType===tc,metalnessMap:at,roughnessMap:pt,anisotropy:De,anisotropyMap:ne,clearcoat:C,clearcoatMap:Ce,clearcoatNormalMap:xe,clearcoatRoughnessMap:ke,dispersion:v,iridescence:z,iridescenceMap:Ve,iridescenceThicknessMap:me,sheen:re,sheenColorMap:pe,sheenRoughnessMap:Te,specularMap:Re,specularColorMap:Ee,specularIntensityMap:He,transmission:le,transmissionMap:k,thicknessMap:Se,gradientMap:J,opaque:S.transparent===!1&&S.blending===Gi&&S.alphaToCoverage===!1,alphaMap:we,alphaTest:ge,alphaHash:te,combine:S.combine,mapUv:Fe&&_(S.map.channel),aoMapUv:ct&&_(S.aoMap.channel),lightMapUv:dt&&_(S.lightMap.channel),bumpMapUv:Ke&&_(S.bumpMap.channel),normalMapUv:St&&_(S.normalMap.channel),displacementMapUv:B&&_(S.displacementMap.channel),emissiveMapUv:Mt&&_(S.emissiveMap.channel),metalnessMapUv:at&&_(S.metalnessMap.channel),roughnessMapUv:pt&&_(S.roughnessMap.channel),anisotropyMapUv:ne&&_(S.anisotropyMap.channel),clearcoatMapUv:Ce&&_(S.clearcoatMap.channel),clearcoatNormalMapUv:xe&&_(S.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:ke&&_(S.clearcoatRoughnessMap.channel),iridescenceMapUv:Ve&&_(S.iridescenceMap.channel),iridescenceThicknessMapUv:me&&_(S.iridescenceThicknessMap.channel),sheenColorMapUv:pe&&_(S.sheenColorMap.channel),sheenRoughnessMapUv:Te&&_(S.sheenRoughnessMap.channel),specularMapUv:Re&&_(S.specularMap.channel),specularColorMapUv:Ee&&_(S.specularColorMap.channel),specularIntensityMapUv:He&&_(S.specularIntensityMap.channel),transmissionMapUv:k&&_(S.transmissionMap.channel),thicknessMapUv:Se&&_(S.thicknessMap.channel),alphaMapUv:we&&_(S.alphaMap.channel),vertexTangents:!!K.attributes.tangent&&(St||De),vertexColors:S.vertexColors,vertexAlphas:S.vertexColors===!0&&!!K.attributes.color&&K.attributes.color.itemSize===4,pointsUvs:O.isPoints===!0&&!!K.attributes.uv&&(Fe||we),fog:!!V,useFog:S.fog===!0,fogExp2:!!V&&V.isFogExp2,flatShading:S.wireframe===!1&&(S.flatShading===!0||K.attributes.normal===void 0&&St===!1&&(S.isMeshLambertMaterial||S.isMeshPhongMaterial||S.isMeshStandardMaterial||S.isMeshPhysicalMaterial)),sizeAttenuation:S.sizeAttenuation===!0,logarithmicDepthBuffer:d,reversedDepthBuffer:de,skinning:O.isSkinnedMesh===!0,morphTargets:K.morphAttributes.position!==void 0,morphNormals:K.morphAttributes.normal!==void 0,morphColors:K.morphAttributes.color!==void 0,morphTargetsCount:Ae,morphTextureStride:ve,numDirLights:T.directional.length,numPointLights:T.point.length,numSpotLights:T.spot.length,numSpotLightMaps:T.spotLightMap.length,numRectAreaLights:T.rectArea.length,numHemiLights:T.hemi.length,numDirLightShadows:T.directionalShadowMap.length,numPointLightShadows:T.pointShadowMap.length,numSpotLightShadows:T.spotShadowMap.length,numSpotLightShadowsWithMaps:T.numSpotLightShadowsWithMaps,numLightProbes:T.numLightProbes,numClippingPlanes:s.numPlanes,numClipIntersection:s.numIntersection,dithering:S.dithering,shadowMapEnabled:i.shadowMap.enabled&&G.length>0,shadowMapType:i.shadowMap.type,toneMapping:qe,decodeVideoTexture:Fe&&S.map.isVideoTexture===!0&&lt.getTransfer(S.map.colorSpace)===ft,decodeVideoTextureEmissive:Mt&&S.emissiveMap.isVideoTexture===!0&&lt.getTransfer(S.emissiveMap.colorSpace)===ft,premultipliedAlpha:S.premultipliedAlpha,doubleSided:S.side===Bn,flipSided:S.side===qt,useDepthPacking:S.depthPacking>=0,depthPacking:S.depthPacking||0,index0AttributeName:S.index0AttributeName,extensionClipCullDistance:Pe&&S.extensions.clipCullDistance===!0&&t.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(Pe&&S.extensions.multiDraw===!0||Le)&&t.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:t.has("KHR_parallel_shader_compile"),customProgramCacheKey:S.customProgramCacheKey()};return mt.vertexUv1s=c.has(1),mt.vertexUv2s=c.has(2),mt.vertexUv3s=c.has(3),c.clear(),mt}function g(S){const T=[];if(S.shaderID?T.push(S.shaderID):(T.push(S.customVertexShaderID),T.push(S.customFragmentShaderID)),S.defines!==void 0)for(const G in S.defines)T.push(G),T.push(S.defines[G]);return S.isRawShaderMaterial===!1&&(m(T,S),b(T,S),T.push(i.outputColorSpace)),T.push(S.customProgramCacheKey),T.join()}function m(S,T){S.push(T.precision),S.push(T.outputColorSpace),S.push(T.envMapMode),S.push(T.envMapCubeUVHeight),S.push(T.mapUv),S.push(T.alphaMapUv),S.push(T.lightMapUv),S.push(T.aoMapUv),S.push(T.bumpMapUv),S.push(T.normalMapUv),S.push(T.displacementMapUv),S.push(T.emissiveMapUv),S.push(T.metalnessMapUv),S.push(T.roughnessMapUv),S.push(T.anisotropyMapUv),S.push(T.clearcoatMapUv),S.push(T.clearcoatNormalMapUv),S.push(T.clearcoatRoughnessMapUv),S.push(T.iridescenceMapUv),S.push(T.iridescenceThicknessMapUv),S.push(T.sheenColorMapUv),S.push(T.sheenRoughnessMapUv),S.push(T.specularMapUv),S.push(T.specularColorMapUv),S.push(T.specularIntensityMapUv),S.push(T.transmissionMapUv),S.push(T.thicknessMapUv),S.push(T.combine),S.push(T.fogExp2),S.push(T.sizeAttenuation),S.push(T.morphTargetsCount),S.push(T.morphAttributeCount),S.push(T.numDirLights),S.push(T.numPointLights),S.push(T.numSpotLights),S.push(T.numSpotLightMaps),S.push(T.numHemiLights),S.push(T.numRectAreaLights),S.push(T.numDirLightShadows),S.push(T.numPointLightShadows),S.push(T.numSpotLightShadows),S.push(T.numSpotLightShadowsWithMaps),S.push(T.numLightProbes),S.push(T.shadowMapType),S.push(T.toneMapping),S.push(T.numClippingPlanes),S.push(T.numClipIntersection),S.push(T.depthPacking)}function b(S,T){a.disableAll(),T.instancing&&a.enable(0),T.instancingColor&&a.enable(1),T.instancingMorph&&a.enable(2),T.matcap&&a.enable(3),T.envMap&&a.enable(4),T.normalMapObjectSpace&&a.enable(5),T.normalMapTangentSpace&&a.enable(6),T.clearcoat&&a.enable(7),T.iridescence&&a.enable(8),T.alphaTest&&a.enable(9),T.vertexColors&&a.enable(10),T.vertexAlphas&&a.enable(11),T.vertexUv1s&&a.enable(12),T.vertexUv2s&&a.enable(13),T.vertexUv3s&&a.enable(14),T.vertexTangents&&a.enable(15),T.anisotropy&&a.enable(16),T.alphaHash&&a.enable(17),T.batching&&a.enable(18),T.dispersion&&a.enable(19),T.batchingColor&&a.enable(20),T.gradientMap&&a.enable(21),S.push(a.mask),a.disableAll(),T.fog&&a.enable(0),T.useFog&&a.enable(1),T.flatShading&&a.enable(2),T.logarithmicDepthBuffer&&a.enable(3),T.reversedDepthBuffer&&a.enable(4),T.skinning&&a.enable(5),T.morphTargets&&a.enable(6),T.morphNormals&&a.enable(7),T.morphColors&&a.enable(8),T.premultipliedAlpha&&a.enable(9),T.shadowMapEnabled&&a.enable(10),T.doubleSided&&a.enable(11),T.flipSided&&a.enable(12),T.useDepthPacking&&a.enable(13),T.dithering&&a.enable(14),T.transmission&&a.enable(15),T.sheen&&a.enable(16),T.opaque&&a.enable(17),T.pointsUvs&&a.enable(18),T.decodeVideoTexture&&a.enable(19),T.decodeVideoTextureEmissive&&a.enable(20),T.alphaToCoverage&&a.enable(21),S.push(a.mask)}function w(S){const T=f[S.type];let G;if(T){const D=bn[T];G=Yh.clone(D.uniforms)}else G=S.uniforms;return G}function A(S,T){let G=u.get(T);return G!==void 0?++G.usedTimes:(G=new Rm(i,T,S,r),l.push(G),u.set(T,G)),G}function U(S){if(--S.usedTimes===0){const T=l.indexOf(S);l[T]=l[l.length-1],l.pop(),u.delete(S.cacheKey),S.destroy()}}function L(S){o.remove(S)}function N(){o.dispose()}return{getParameters:y,getProgramCacheKey:g,getUniforms:w,acquireProgram:A,releaseProgram:U,releaseShaderCache:L,programs:l,dispose:N}}function Um(){let i=new WeakMap;function e(a){return i.has(a)}function t(a){let o=i.get(a);return o===void 0&&(o={},i.set(a,o)),o}function n(a){i.delete(a)}function r(a,o,c){i.get(a)[o]=c}function s(){i=new WeakMap}return{has:e,get:t,remove:n,update:r,dispose:s}}function Fm(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.material.id!==e.material.id?i.material.id-e.material.id:i.materialVariant!==e.materialVariant?i.materialVariant-e.materialVariant:i.z!==e.z?i.z-e.z:i.id-e.id}function Pl(i,e){return i.groupOrder!==e.groupOrder?i.groupOrder-e.groupOrder:i.renderOrder!==e.renderOrder?i.renderOrder-e.renderOrder:i.z!==e.z?e.z-i.z:i.id-e.id}function Dl(){const i=[];let e=0;const t=[],n=[],r=[];function s(){e=0,t.length=0,n.length=0,r.length=0}function a(h){let f=0;return h.isInstancedMesh&&(f+=2),h.isSkinnedMesh&&(f+=1),f}function o(h,f,_,y,g,m){let b=i[e];return b===void 0?(b={id:h.id,object:h,geometry:f,material:_,materialVariant:a(h),groupOrder:y,renderOrder:h.renderOrder,z:g,group:m},i[e]=b):(b.id=h.id,b.object=h,b.geometry=f,b.material=_,b.materialVariant=a(h),b.groupOrder=y,b.renderOrder=h.renderOrder,b.z=g,b.group=m),e++,b}function c(h,f,_,y,g,m){const b=o(h,f,_,y,g,m);_.transmission>0?n.push(b):_.transparent===!0?r.push(b):t.push(b)}function l(h,f,_,y,g,m){const b=o(h,f,_,y,g,m);_.transmission>0?n.unshift(b):_.transparent===!0?r.unshift(b):t.unshift(b)}function u(h,f){t.length>1&&t.sort(h||Fm),n.length>1&&n.sort(f||Pl),r.length>1&&r.sort(f||Pl)}function d(){for(let h=e,f=i.length;h<f;h++){const _=i[h];if(_.id===null)break;_.id=null,_.object=null,_.geometry=null,_.material=null,_.group=null}}return{opaque:t,transmissive:n,transparent:r,init:s,push:c,unshift:l,finish:d,sort:u}}function Nm(){let i=new WeakMap;function e(n,r){const s=i.get(n);let a;return s===void 0?(a=new Dl,i.set(n,[a])):r>=s.length?(a=new Dl,s.push(a)):a=s[r],a}function t(){i=new WeakMap}return{get:e,dispose:t}}function Om(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={direction:new q,color:new rt};break;case"SpotLight":t={position:new q,direction:new q,color:new rt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":t={position:new q,color:new rt,distance:0,decay:0};break;case"HemisphereLight":t={direction:new q,skyColor:new rt,groundColor:new rt};break;case"RectAreaLight":t={color:new rt,position:new q,halfWidth:new q,halfHeight:new q};break}return i[e.id]=t,t}}}function Bm(){const i={};return{get:function(e){if(i[e.id]!==void 0)return i[e.id];let t;switch(e.type){case"DirectionalLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new $e};break;case"SpotLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new $e};break;case"PointLight":t={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new $e,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[e.id]=t,t}}}let km=0;function zm(i,e){return(e.castShadow?2:0)-(i.castShadow?2:0)+(e.map?1:0)-(i.map?1:0)}function Gm(i){const e=new Om,t=Bm(),n={version:0,hash:{directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)n.probe.push(new q);const r=new q,s=new _t,a=new _t;function o(l){let u=0,d=0,h=0;for(let T=0;T<9;T++)n.probe[T].set(0,0,0);let f=0,_=0,y=0,g=0,m=0,b=0,w=0,A=0,U=0,L=0,N=0;l.sort(zm);for(let T=0,G=l.length;T<G;T++){const D=l[T],O=D.color,V=D.intensity,K=D.distance;let Y=null;if(D.shadow&&D.shadow.map&&(D.shadow.map.texture.format===mn?Y=D.shadow.map.texture:Y=D.shadow.map.depthTexture||D.shadow.map.texture),D.isAmbientLight)u+=O.r*V,d+=O.g*V,h+=O.b*V;else if(D.isLightProbe){for(let Z=0;Z<9;Z++)n.probe[Z].addScaledVector(D.sh.coefficients[Z],V);N++}else if(D.isDirectionalLight){const Z=e.get(D);if(Z.color.copy(D.color).multiplyScalar(D.intensity),D.castShadow){const X=D.shadow,fe=t.get(D);fe.shadowIntensity=X.intensity,fe.shadowBias=X.bias,fe.shadowNormalBias=X.normalBias,fe.shadowRadius=X.radius,fe.shadowMapSize=X.mapSize,n.directionalShadow[f]=fe,n.directionalShadowMap[f]=Y,n.directionalShadowMatrix[f]=D.shadow.matrix,b++}n.directional[f]=Z,f++}else if(D.isSpotLight){const Z=e.get(D);Z.position.setFromMatrixPosition(D.matrixWorld),Z.color.copy(O).multiplyScalar(V),Z.distance=K,Z.coneCos=Math.cos(D.angle),Z.penumbraCos=Math.cos(D.angle*(1-D.penumbra)),Z.decay=D.decay,n.spot[y]=Z;const X=D.shadow;if(D.map&&(n.spotLightMap[U]=D.map,U++,X.updateMatrices(D),D.castShadow&&L++),n.spotLightMatrix[y]=X.matrix,D.castShadow){const fe=t.get(D);fe.shadowIntensity=X.intensity,fe.shadowBias=X.bias,fe.shadowNormalBias=X.normalBias,fe.shadowRadius=X.radius,fe.shadowMapSize=X.mapSize,n.spotShadow[y]=fe,n.spotShadowMap[y]=Y,A++}y++}else if(D.isRectAreaLight){const Z=e.get(D);Z.color.copy(O).multiplyScalar(V),Z.halfWidth.set(D.width*.5,0,0),Z.halfHeight.set(0,D.height*.5,0),n.rectArea[g]=Z,g++}else if(D.isPointLight){const Z=e.get(D);if(Z.color.copy(D.color).multiplyScalar(D.intensity),Z.distance=D.distance,Z.decay=D.decay,D.castShadow){const X=D.shadow,fe=t.get(D);fe.shadowIntensity=X.intensity,fe.shadowBias=X.bias,fe.shadowNormalBias=X.normalBias,fe.shadowRadius=X.radius,fe.shadowMapSize=X.mapSize,fe.shadowCameraNear=X.camera.near,fe.shadowCameraFar=X.camera.far,n.pointShadow[_]=fe,n.pointShadowMap[_]=Y,n.pointShadowMatrix[_]=D.shadow.matrix,w++}n.point[_]=Z,_++}else if(D.isHemisphereLight){const Z=e.get(D);Z.skyColor.copy(D.color).multiplyScalar(V),Z.groundColor.copy(D.groundColor).multiplyScalar(V),n.hemi[m]=Z,m++}}g>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=be.LTC_FLOAT_1,n.rectAreaLTC2=be.LTC_FLOAT_2):(n.rectAreaLTC1=be.LTC_HALF_1,n.rectAreaLTC2=be.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=d,n.ambient[2]=h;const S=n.hash;(S.directionalLength!==f||S.pointLength!==_||S.spotLength!==y||S.rectAreaLength!==g||S.hemiLength!==m||S.numDirectionalShadows!==b||S.numPointShadows!==w||S.numSpotShadows!==A||S.numSpotMaps!==U||S.numLightProbes!==N)&&(n.directional.length=f,n.spot.length=y,n.rectArea.length=g,n.point.length=_,n.hemi.length=m,n.directionalShadow.length=b,n.directionalShadowMap.length=b,n.pointShadow.length=w,n.pointShadowMap.length=w,n.spotShadow.length=A,n.spotShadowMap.length=A,n.directionalShadowMatrix.length=b,n.pointShadowMatrix.length=w,n.spotLightMatrix.length=A+U-L,n.spotLightMap.length=U,n.numSpotLightShadowsWithMaps=L,n.numLightProbes=N,S.directionalLength=f,S.pointLength=_,S.spotLength=y,S.rectAreaLength=g,S.hemiLength=m,S.numDirectionalShadows=b,S.numPointShadows=w,S.numSpotShadows=A,S.numSpotMaps=U,S.numLightProbes=N,n.version=km++)}function c(l,u){let d=0,h=0,f=0,_=0,y=0;const g=u.matrixWorldInverse;for(let m=0,b=l.length;m<b;m++){const w=l[m];if(w.isDirectionalLight){const A=n.directional[d];A.direction.setFromMatrixPosition(w.matrixWorld),r.setFromMatrixPosition(w.target.matrixWorld),A.direction.sub(r),A.direction.transformDirection(g),d++}else if(w.isSpotLight){const A=n.spot[f];A.position.setFromMatrixPosition(w.matrixWorld),A.position.applyMatrix4(g),A.direction.setFromMatrixPosition(w.matrixWorld),r.setFromMatrixPosition(w.target.matrixWorld),A.direction.sub(r),A.direction.transformDirection(g),f++}else if(w.isRectAreaLight){const A=n.rectArea[_];A.position.setFromMatrixPosition(w.matrixWorld),A.position.applyMatrix4(g),a.identity(),s.copy(w.matrixWorld),s.premultiply(g),a.extractRotation(s),A.halfWidth.set(w.width*.5,0,0),A.halfHeight.set(0,w.height*.5,0),A.halfWidth.applyMatrix4(a),A.halfHeight.applyMatrix4(a),_++}else if(w.isPointLight){const A=n.point[h];A.position.setFromMatrixPosition(w.matrixWorld),A.position.applyMatrix4(g),h++}else if(w.isHemisphereLight){const A=n.hemi[y];A.direction.setFromMatrixPosition(w.matrixWorld),A.direction.transformDirection(g),y++}}}return{setup:o,setupView:c,state:n}}function Il(i){const e=new Gm(i),t=[],n=[];function r(u){l.camera=u,t.length=0,n.length=0}function s(u){t.push(u)}function a(u){n.push(u)}function o(){e.setup(t)}function c(u){e.setupView(t,u)}const l={lightsArray:t,shadowsArray:n,camera:null,lights:e,transmissionRenderTarget:{}};return{init:r,state:l,setupLights:o,setupLightsView:c,pushLight:s,pushShadow:a}}function Hm(i){let e=new WeakMap;function t(r,s=0){const a=e.get(r);let o;return a===void 0?(o=new Il(i),e.set(r,[o])):s>=a.length?(o=new Il(i),a.push(o)):o=a[s],o}function n(){e=new WeakMap}return{get:t,dispose:n}}const Vm=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,Wm=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,Xm=[new q(1,0,0),new q(-1,0,0),new q(0,1,0),new q(0,-1,0),new q(0,0,1),new q(0,0,-1)],Ym=[new q(0,-1,0),new q(0,-1,0),new q(0,0,1),new q(0,0,-1),new q(0,-1,0),new q(0,-1,0)],Ll=new _t,or=new q,ta=new q;function qm(i,e,t){let n=new _o;const r=new $e,s=new $e,a=new Et,o=new Kh,c=new Jh,l={},u=t.maxTextureSize,d={[ri]:qt,[qt]:ri,[Bn]:Bn},h=new Pn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new $e},radius:{value:4}},vertexShader:Vm,fragmentShader:Wm}),f=h.clone();f.defines.HORIZONTAL_PASS=1;const _=new Qt;_.setAttribute("position",new on(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));const y=new _n(_,h),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=rs;let m=this.type;this.render=function(L,N,S){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||L.length===0)return;this.type===Dc&&(Xe("WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead."),this.type=rs);const T=i.getRenderTarget(),G=i.getActiveCubeFace(),D=i.getActiveMipmapLevel(),O=i.state;O.setBlending(zn),O.buffers.depth.getReversed()===!0?O.buffers.color.setClear(0,0,0,0):O.buffers.color.setClear(1,1,1,1),O.buffers.depth.setTest(!0),O.setScissorTest(!1);const V=m!==this.type;V&&N.traverse(function(K){K.material&&(Array.isArray(K.material)?K.material.forEach(Y=>Y.needsUpdate=!0):K.material.needsUpdate=!0)});for(let K=0,Y=L.length;K<Y;K++){const Z=L[K],X=Z.shadow;if(X===void 0){Xe("WebGLShadowMap:",Z,"has no shadow.");continue}if(X.autoUpdate===!1&&X.needsUpdate===!1)continue;r.copy(X.mapSize);const fe=X.getFrameExtents();r.multiply(fe),s.copy(X.mapSize),(r.x>u||r.y>u)&&(r.x>u&&(s.x=Math.floor(u/fe.x),r.x=s.x*fe.x,X.mapSize.x=s.x),r.y>u&&(s.y=Math.floor(u/fe.y),r.y=s.y*fe.y,X.mapSize.y=s.y));const oe=i.state.buffers.depth.getReversed();if(X.camera._reversedDepth=oe,X.map===null||V===!0){if(X.map!==null&&(X.map.depthTexture!==null&&(X.map.depthTexture.dispose(),X.map.depthTexture=null),X.map.dispose()),this.type===ur){if(Z.isPointLight){Xe("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}X.map=new wn(r.x,r.y,{format:mn,type:Jt,minFilter:bt,magFilter:bt,generateMipmaps:!1}),X.map.texture.name=Z.name+".shadowMap",X.map.depthTexture=new xr(r.x,r.y,Yt),X.map.depthTexture.name=Z.name+".shadowMapDepth",X.map.depthTexture.format=Hn,X.map.depthTexture.compareFunction=null,X.map.depthTexture.minFilter=Nt,X.map.depthTexture.magFilter=Nt}else Z.isPointLight?(X.map=new mc(r.x),X.map.depthTexture=new Vh(r.x,Cn)):(X.map=new wn(r.x,r.y),X.map.depthTexture=new xr(r.x,r.y,Cn)),X.map.depthTexture.name=Z.name+".shadowMap",X.map.depthTexture.format=Hn,this.type===rs?(X.map.depthTexture.compareFunction=oe?fo:uo,X.map.depthTexture.minFilter=bt,X.map.depthTexture.magFilter=bt):(X.map.depthTexture.compareFunction=null,X.map.depthTexture.minFilter=Nt,X.map.depthTexture.magFilter=Nt);X.camera.updateProjectionMatrix()}const ye=X.map.isWebGLCubeRenderTarget?6:1;for(let Ae=0;Ae<ye;Ae++){if(X.map.isWebGLCubeRenderTarget)i.setRenderTarget(X.map,Ae),i.clear();else{Ae===0&&(i.setRenderTarget(X.map),i.clear());const ve=X.getViewport(Ae);a.set(s.x*ve.x,s.y*ve.y,s.x*ve.z,s.y*ve.w),O.viewport(a)}if(Z.isPointLight){const ve=X.camera,Ge=X.matrix,st=Z.distance||ve.far;st!==ve.far&&(ve.far=st,ve.updateProjectionMatrix()),or.setFromMatrixPosition(Z.matrixWorld),ve.position.copy(or),ta.copy(ve.position),ta.add(Xm[Ae]),ve.up.copy(Ym[Ae]),ve.lookAt(ta),ve.updateMatrixWorld(),Ge.makeTranslation(-or.x,-or.y,-or.z),Ll.multiplyMatrices(ve.projectionMatrix,ve.matrixWorldInverse),X._frustum.setFromProjectionMatrix(Ll,ve.coordinateSystem,ve.reversedDepth)}else X.updateMatrices(Z);n=X.getFrustum(),A(N,S,X.camera,Z,this.type)}X.isPointLightShadow!==!0&&this.type===ur&&b(X,S),X.needsUpdate=!1}m=this.type,g.needsUpdate=!1,i.setRenderTarget(T,G,D)};function b(L,N){const S=e.update(y);h.defines.VSM_SAMPLES!==L.blurSamples&&(h.defines.VSM_SAMPLES=L.blurSamples,f.defines.VSM_SAMPLES=L.blurSamples,h.needsUpdate=!0,f.needsUpdate=!0),L.mapPass===null&&(L.mapPass=new wn(r.x,r.y,{format:mn,type:Jt})),h.uniforms.shadow_pass.value=L.map.depthTexture,h.uniforms.resolution.value=L.mapSize,h.uniforms.radius.value=L.radius,i.setRenderTarget(L.mapPass),i.clear(),i.renderBufferDirect(N,null,S,h,y,null),f.uniforms.shadow_pass.value=L.mapPass.texture,f.uniforms.resolution.value=L.mapSize,f.uniforms.radius.value=L.radius,i.setRenderTarget(L.map),i.clear(),i.renderBufferDirect(N,null,S,f,y,null)}function w(L,N,S,T){let G=null;const D=S.isPointLight===!0?L.customDistanceMaterial:L.customDepthMaterial;if(D!==void 0)G=D;else if(G=S.isPointLight===!0?c:o,i.localClippingEnabled&&N.clipShadows===!0&&Array.isArray(N.clippingPlanes)&&N.clippingPlanes.length!==0||N.displacementMap&&N.displacementScale!==0||N.alphaMap&&N.alphaTest>0||N.map&&N.alphaTest>0||N.alphaToCoverage===!0){const O=G.uuid,V=N.uuid;let K=l[O];K===void 0&&(K={},l[O]=K);let Y=K[V];Y===void 0&&(Y=G.clone(),K[V]=Y,N.addEventListener("dispose",U)),G=Y}if(G.visible=N.visible,G.wireframe=N.wireframe,T===ur?G.side=N.shadowSide!==null?N.shadowSide:N.side:G.side=N.shadowSide!==null?N.shadowSide:d[N.side],G.alphaMap=N.alphaMap,G.alphaTest=N.alphaToCoverage===!0?.5:N.alphaTest,G.map=N.map,G.clipShadows=N.clipShadows,G.clippingPlanes=N.clippingPlanes,G.clipIntersection=N.clipIntersection,G.displacementMap=N.displacementMap,G.displacementScale=N.displacementScale,G.displacementBias=N.displacementBias,G.wireframeLinewidth=N.wireframeLinewidth,G.linewidth=N.linewidth,S.isPointLight===!0&&G.isMeshDistanceMaterial===!0){const O=i.properties.get(G);O.light=S}return G}function A(L,N,S,T,G){if(L.visible===!1)return;if(L.layers.test(N.layers)&&(L.isMesh||L.isLine||L.isPoints)&&(L.castShadow||L.receiveShadow&&G===ur)&&(!L.frustumCulled||n.intersectsObject(L))){L.modelViewMatrix.multiplyMatrices(S.matrixWorldInverse,L.matrixWorld);const V=e.update(L),K=L.material;if(Array.isArray(K)){const Y=V.groups;for(let Z=0,X=Y.length;Z<X;Z++){const fe=Y[Z],oe=K[fe.materialIndex];if(oe&&oe.visible){const ye=w(L,oe,T,G);L.onBeforeShadow(i,L,N,S,V,ye,fe),i.renderBufferDirect(S,null,V,ye,L,fe),L.onAfterShadow(i,L,N,S,V,ye,fe)}}}else if(K.visible){const Y=w(L,K,T,G);L.onBeforeShadow(i,L,N,S,V,Y,null),i.renderBufferDirect(S,null,V,Y,L,null),L.onAfterShadow(i,L,N,S,V,Y,null)}}const O=L.children;for(let V=0,K=O.length;V<K;V++)A(O[V],N,S,T,G)}function U(L){L.target.removeEventListener("dispose",U);for(const S in l){const T=l[S],G=L.target.uuid;G in T&&(T[G].dispose(),delete T[G])}}}function Zm(i,e){function t(){let k=!1;const Se=new Et;let J=null;const we=new Et(0,0,0,0);return{setMask:function(ge){J!==ge&&!k&&(i.colorMask(ge,ge,ge,ge),J=ge)},setLocked:function(ge){k=ge},setClear:function(ge,te,Pe,qe,mt){mt===!0&&(ge*=qe,te*=qe,Pe*=qe),Se.set(ge,te,Pe,qe),we.equals(Se)===!1&&(i.clearColor(ge,te,Pe,qe),we.copy(Se))},reset:function(){k=!1,J=null,we.set(-1,0,0,0)}}}function n(){let k=!1,Se=!1,J=null,we=null,ge=null;return{setReversed:function(te){if(Se!==te){const Pe=e.get("EXT_clip_control");te?Pe.clipControlEXT(Pe.LOWER_LEFT_EXT,Pe.ZERO_TO_ONE_EXT):Pe.clipControlEXT(Pe.LOWER_LEFT_EXT,Pe.NEGATIVE_ONE_TO_ONE_EXT),Se=te;const qe=ge;ge=null,this.setClear(qe)}},getReversed:function(){return Se},setTest:function(te){te?ue(i.DEPTH_TEST):de(i.DEPTH_TEST)},setMask:function(te){J!==te&&!k&&(i.depthMask(te),J=te)},setFunc:function(te){if(Se&&(te=uh[te]),we!==te){switch(te){case ca:i.depthFunc(i.NEVER);break;case ha:i.depthFunc(i.ALWAYS);break;case ua:i.depthFunc(i.LESS);break;case Vi:i.depthFunc(i.LEQUAL);break;case da:i.depthFunc(i.EQUAL);break;case fa:i.depthFunc(i.GEQUAL);break;case pa:i.depthFunc(i.GREATER);break;case ma:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}we=te}},setLocked:function(te){k=te},setClear:function(te){ge!==te&&(ge=te,Se&&(te=1-te),i.clearDepth(te))},reset:function(){k=!1,J=null,we=null,ge=null,Se=!1}}}function r(){let k=!1,Se=null,J=null,we=null,ge=null,te=null,Pe=null,qe=null,mt=null;return{setTest:function(ut){k||(ut?ue(i.STENCIL_TEST):de(i.STENCIL_TEST))},setMask:function(ut){Se!==ut&&!k&&(i.stencilMask(ut),Se=ut)},setFunc:function(ut,ln,cn){(J!==ut||we!==ln||ge!==cn)&&(i.stencilFunc(ut,ln,cn),J=ut,we=ln,ge=cn)},setOp:function(ut,ln,cn){(te!==ut||Pe!==ln||qe!==cn)&&(i.stencilOp(ut,ln,cn),te=ut,Pe=ln,qe=cn)},setLocked:function(ut){k=ut},setClear:function(ut){mt!==ut&&(i.clearStencil(ut),mt=ut)},reset:function(){k=!1,Se=null,J=null,we=null,ge=null,te=null,Pe=null,qe=null,mt=null}}}const s=new t,a=new n,o=new r,c=new WeakMap,l=new WeakMap;let u={},d={},h=new WeakMap,f=[],_=null,y=!1,g=null,m=null,b=null,w=null,A=null,U=null,L=null,N=new rt(0,0,0),S=0,T=!1,G=null,D=null,O=null,V=null,K=null;const Y=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS);let Z=!1,X=0;const fe=i.getParameter(i.VERSION);fe.indexOf("WebGL")!==-1?(X=parseFloat(/^WebGL (\d)/.exec(fe)[1]),Z=X>=1):fe.indexOf("OpenGL ES")!==-1&&(X=parseFloat(/^OpenGL ES (\d)/.exec(fe)[1]),Z=X>=2);let oe=null,ye={};const Ae=i.getParameter(i.SCISSOR_BOX),ve=i.getParameter(i.VIEWPORT),Ge=new Et().fromArray(Ae),st=new Et().fromArray(ve);function _e(k,Se,J,we){const ge=new Uint8Array(4),te=i.createTexture();i.bindTexture(k,te),i.texParameteri(k,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(k,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let Pe=0;Pe<J;Pe++)k===i.TEXTURE_3D||k===i.TEXTURE_2D_ARRAY?i.texImage3D(Se,0,i.RGBA,1,1,we,0,i.RGBA,i.UNSIGNED_BYTE,ge):i.texImage2D(Se+Pe,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,ge);return te}const $={};$[i.TEXTURE_2D]=_e(i.TEXTURE_2D,i.TEXTURE_2D,1),$[i.TEXTURE_CUBE_MAP]=_e(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),$[i.TEXTURE_2D_ARRAY]=_e(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),$[i.TEXTURE_3D]=_e(i.TEXTURE_3D,i.TEXTURE_3D,1,1),s.setClear(0,0,0,1),a.setClear(1),o.setClear(0),ue(i.DEPTH_TEST),a.setFunc(Vi),Ke(!1),St(bo),ue(i.CULL_FACE),ct(zn);function ue(k){u[k]!==!0&&(i.enable(k),u[k]=!0)}function de(k){u[k]!==!1&&(i.disable(k),u[k]=!1)}function ze(k,Se){return d[k]!==Se?(i.bindFramebuffer(k,Se),d[k]=Se,k===i.DRAW_FRAMEBUFFER&&(d[i.FRAMEBUFFER]=Se),k===i.FRAMEBUFFER&&(d[i.DRAW_FRAMEBUFFER]=Se),!0):!1}function Le(k,Se){let J=f,we=!1;if(k){J=h.get(Se),J===void 0&&(J=[],h.set(Se,J));const ge=k.textures;if(J.length!==ge.length||J[0]!==i.COLOR_ATTACHMENT0){for(let te=0,Pe=ge.length;te<Pe;te++)J[te]=i.COLOR_ATTACHMENT0+te;J.length=ge.length,we=!0}}else J[0]!==i.BACK&&(J[0]=i.BACK,we=!0);we&&i.drawBuffers(J)}function Fe(k){return _!==k?(i.useProgram(k),_=k,!0):!1}const xt={[fi]:i.FUNC_ADD,[Lc]:i.FUNC_SUBTRACT,[Uc]:i.FUNC_REVERSE_SUBTRACT};xt[Fc]=i.MIN,xt[Nc]=i.MAX;const et={[Oc]:i.ZERO,[Bc]:i.ONE,[kc]:i.SRC_COLOR,[oa]:i.SRC_ALPHA,[Xc]:i.SRC_ALPHA_SATURATE,[Vc]:i.DST_COLOR,[Gc]:i.DST_ALPHA,[zc]:i.ONE_MINUS_SRC_COLOR,[la]:i.ONE_MINUS_SRC_ALPHA,[Wc]:i.ONE_MINUS_DST_COLOR,[Hc]:i.ONE_MINUS_DST_ALPHA,[Yc]:i.CONSTANT_COLOR,[qc]:i.ONE_MINUS_CONSTANT_COLOR,[Zc]:i.CONSTANT_ALPHA,[$c]:i.ONE_MINUS_CONSTANT_ALPHA};function ct(k,Se,J,we,ge,te,Pe,qe,mt,ut){if(k===zn){y===!0&&(de(i.BLEND),y=!1);return}if(y===!1&&(ue(i.BLEND),y=!0),k!==Ic){if(k!==g||ut!==T){if((m!==fi||A!==fi)&&(i.blendEquation(i.FUNC_ADD),m=fi,A=fi),ut)switch(k){case Gi:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case To:i.blendFunc(i.ONE,i.ONE);break;case Ao:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case wo:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:ot("WebGLState: Invalid blending: ",k);break}else switch(k){case Gi:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case To:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case Ao:ot("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case wo:ot("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:ot("WebGLState: Invalid blending: ",k);break}b=null,w=null,U=null,L=null,N.set(0,0,0),S=0,g=k,T=ut}return}ge=ge||Se,te=te||J,Pe=Pe||we,(Se!==m||ge!==A)&&(i.blendEquationSeparate(xt[Se],xt[ge]),m=Se,A=ge),(J!==b||we!==w||te!==U||Pe!==L)&&(i.blendFuncSeparate(et[J],et[we],et[te],et[Pe]),b=J,w=we,U=te,L=Pe),(qe.equals(N)===!1||mt!==S)&&(i.blendColor(qe.r,qe.g,qe.b,mt),N.copy(qe),S=mt),g=k,T=!1}function dt(k,Se){k.side===Bn?de(i.CULL_FACE):ue(i.CULL_FACE);let J=k.side===qt;Se&&(J=!J),Ke(J),k.blending===Gi&&k.transparent===!1?ct(zn):ct(k.blending,k.blendEquation,k.blendSrc,k.blendDst,k.blendEquationAlpha,k.blendSrcAlpha,k.blendDstAlpha,k.blendColor,k.blendAlpha,k.premultipliedAlpha),a.setFunc(k.depthFunc),a.setTest(k.depthTest),a.setMask(k.depthWrite),s.setMask(k.colorWrite);const we=k.stencilWrite;o.setTest(we),we&&(o.setMask(k.stencilWriteMask),o.setFunc(k.stencilFunc,k.stencilRef,k.stencilFuncMask),o.setOp(k.stencilFail,k.stencilZFail,k.stencilZPass)),Mt(k.polygonOffset,k.polygonOffsetFactor,k.polygonOffsetUnits),k.alphaToCoverage===!0?ue(i.SAMPLE_ALPHA_TO_COVERAGE):de(i.SAMPLE_ALPHA_TO_COVERAGE)}function Ke(k){G!==k&&(k?i.frontFace(i.CW):i.frontFace(i.CCW),G=k)}function St(k){k!==Rc?(ue(i.CULL_FACE),k!==D&&(k===bo?i.cullFace(i.BACK):k===Pc?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):de(i.CULL_FACE),D=k}function B(k){k!==O&&(Z&&i.lineWidth(k),O=k)}function Mt(k,Se,J){k?(ue(i.POLYGON_OFFSET_FILL),(V!==Se||K!==J)&&(V=Se,K=J,a.getReversed()&&(Se=-Se),i.polygonOffset(Se,J))):de(i.POLYGON_OFFSET_FILL)}function at(k){k?ue(i.SCISSOR_TEST):de(i.SCISSOR_TEST)}function pt(k){k===void 0&&(k=i.TEXTURE0+Y-1),oe!==k&&(i.activeTexture(k),oe=k)}function De(k,Se,J){J===void 0&&(oe===null?J=i.TEXTURE0+Y-1:J=oe);let we=ye[J];we===void 0&&(we={type:void 0,texture:void 0},ye[J]=we),(we.type!==k||we.texture!==Se)&&(oe!==J&&(i.activeTexture(J),oe=J),i.bindTexture(k,Se||$[k]),we.type=k,we.texture=Se)}function C(){const k=ye[oe];k!==void 0&&k.type!==void 0&&(i.bindTexture(k.type,null),k.type=void 0,k.texture=void 0)}function v(){try{i.compressedTexImage2D(...arguments)}catch(k){ot("WebGLState:",k)}}function z(){try{i.compressedTexImage3D(...arguments)}catch(k){ot("WebGLState:",k)}}function re(){try{i.texSubImage2D(...arguments)}catch(k){ot("WebGLState:",k)}}function le(){try{i.texSubImage3D(...arguments)}catch(k){ot("WebGLState:",k)}}function ne(){try{i.compressedTexSubImage2D(...arguments)}catch(k){ot("WebGLState:",k)}}function Ce(){try{i.compressedTexSubImage3D(...arguments)}catch(k){ot("WebGLState:",k)}}function xe(){try{i.texStorage2D(...arguments)}catch(k){ot("WebGLState:",k)}}function ke(){try{i.texStorage3D(...arguments)}catch(k){ot("WebGLState:",k)}}function Ve(){try{i.texImage2D(...arguments)}catch(k){ot("WebGLState:",k)}}function me(){try{i.texImage3D(...arguments)}catch(k){ot("WebGLState:",k)}}function pe(k){Ge.equals(k)===!1&&(i.scissor(k.x,k.y,k.z,k.w),Ge.copy(k))}function Te(k){st.equals(k)===!1&&(i.viewport(k.x,k.y,k.z,k.w),st.copy(k))}function Re(k,Se){let J=l.get(Se);J===void 0&&(J=new WeakMap,l.set(Se,J));let we=J.get(k);we===void 0&&(we=i.getUniformBlockIndex(Se,k.name),J.set(k,we))}function Ee(k,Se){const we=l.get(Se).get(k);c.get(Se)!==we&&(i.uniformBlockBinding(Se,we,k.__bindingPointIndex),c.set(Se,we))}function He(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),a.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),u={},oe=null,ye={},d={},h=new WeakMap,f=[],_=null,y=!1,g=null,m=null,b=null,w=null,A=null,U=null,L=null,N=new rt(0,0,0),S=0,T=!1,G=null,D=null,O=null,V=null,K=null,Ge.set(0,0,i.canvas.width,i.canvas.height),st.set(0,0,i.canvas.width,i.canvas.height),s.reset(),a.reset(),o.reset()}return{buffers:{color:s,depth:a,stencil:o},enable:ue,disable:de,bindFramebuffer:ze,drawBuffers:Le,useProgram:Fe,setBlending:ct,setMaterial:dt,setFlipSided:Ke,setCullFace:St,setLineWidth:B,setPolygonOffset:Mt,setScissorTest:at,activeTexture:pt,bindTexture:De,unbindTexture:C,compressedTexImage2D:v,compressedTexImage3D:z,texImage2D:Ve,texImage3D:me,updateUBOMapping:Re,uniformBlockBinding:Ee,texStorage2D:xe,texStorage3D:ke,texSubImage2D:re,texSubImage3D:le,compressedTexSubImage2D:ne,compressedTexSubImage3D:Ce,scissor:pe,viewport:Te,reset:He}}function $m(i,e,t,n,r,s,a){const o=e.has("WEBGL_multisampled_render_to_texture")?e.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new $e,u=new WeakMap;let d;const h=new WeakMap;let f=!1;try{f=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function _(C,v){return f?new OffscreenCanvas(C,v):ds("canvas")}function y(C,v,z){let re=1;const le=De(C);if((le.width>z||le.height>z)&&(re=z/Math.max(le.width,le.height)),re<1)if(typeof HTMLImageElement<"u"&&C instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&C instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&C instanceof ImageBitmap||typeof VideoFrame<"u"&&C instanceof VideoFrame){const ne=Math.floor(re*le.width),Ce=Math.floor(re*le.height);d===void 0&&(d=_(ne,Ce));const xe=v?_(ne,Ce):d;return xe.width=ne,xe.height=Ce,xe.getContext("2d").drawImage(C,0,0,ne,Ce),Xe("WebGLRenderer: Texture has been resized from ("+le.width+"x"+le.height+") to ("+ne+"x"+Ce+")."),xe}else return"data"in C&&Xe("WebGLRenderer: Image in DataTexture is too big ("+le.width+"x"+le.height+")."),C;return C}function g(C){return C.generateMipmaps}function m(C){i.generateMipmap(C)}function b(C){return C.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:C.isWebGL3DRenderTarget?i.TEXTURE_3D:C.isWebGLArrayRenderTarget||C.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function w(C,v,z,re,le=!1){if(C!==null){if(i[C]!==void 0)return i[C];Xe("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+C+"'")}let ne=v;if(v===i.RED&&(z===i.FLOAT&&(ne=i.R32F),z===i.HALF_FLOAT&&(ne=i.R16F),z===i.UNSIGNED_BYTE&&(ne=i.R8)),v===i.RED_INTEGER&&(z===i.UNSIGNED_BYTE&&(ne=i.R8UI),z===i.UNSIGNED_SHORT&&(ne=i.R16UI),z===i.UNSIGNED_INT&&(ne=i.R32UI),z===i.BYTE&&(ne=i.R8I),z===i.SHORT&&(ne=i.R16I),z===i.INT&&(ne=i.R32I)),v===i.RG&&(z===i.FLOAT&&(ne=i.RG32F),z===i.HALF_FLOAT&&(ne=i.RG16F),z===i.UNSIGNED_BYTE&&(ne=i.RG8)),v===i.RG_INTEGER&&(z===i.UNSIGNED_BYTE&&(ne=i.RG8UI),z===i.UNSIGNED_SHORT&&(ne=i.RG16UI),z===i.UNSIGNED_INT&&(ne=i.RG32UI),z===i.BYTE&&(ne=i.RG8I),z===i.SHORT&&(ne=i.RG16I),z===i.INT&&(ne=i.RG32I)),v===i.RGB_INTEGER&&(z===i.UNSIGNED_BYTE&&(ne=i.RGB8UI),z===i.UNSIGNED_SHORT&&(ne=i.RGB16UI),z===i.UNSIGNED_INT&&(ne=i.RGB32UI),z===i.BYTE&&(ne=i.RGB8I),z===i.SHORT&&(ne=i.RGB16I),z===i.INT&&(ne=i.RGB32I)),v===i.RGBA_INTEGER&&(z===i.UNSIGNED_BYTE&&(ne=i.RGBA8UI),z===i.UNSIGNED_SHORT&&(ne=i.RGBA16UI),z===i.UNSIGNED_INT&&(ne=i.RGBA32UI),z===i.BYTE&&(ne=i.RGBA8I),z===i.SHORT&&(ne=i.RGBA16I),z===i.INT&&(ne=i.RGBA32I)),v===i.RGB&&(z===i.UNSIGNED_INT_5_9_9_9_REV&&(ne=i.RGB9_E5),z===i.UNSIGNED_INT_10F_11F_11F_REV&&(ne=i.R11F_G11F_B10F)),v===i.RGBA){const Ce=le?us:lt.getTransfer(re);z===i.FLOAT&&(ne=i.RGBA32F),z===i.HALF_FLOAT&&(ne=i.RGBA16F),z===i.UNSIGNED_BYTE&&(ne=Ce===ft?i.SRGB8_ALPHA8:i.RGBA8),z===i.UNSIGNED_SHORT_4_4_4_4&&(ne=i.RGBA4),z===i.UNSIGNED_SHORT_5_5_5_1&&(ne=i.RGB5_A1)}return(ne===i.R16F||ne===i.R32F||ne===i.RG16F||ne===i.RG32F||ne===i.RGBA16F||ne===i.RGBA32F)&&e.get("EXT_color_buffer_float"),ne}function A(C,v){let z;return C?v===null||v===Cn||v===gr?z=i.DEPTH24_STENCIL8:v===Yt?z=i.DEPTH32F_STENCIL8:v===mr&&(z=i.DEPTH24_STENCIL8,Xe("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):v===null||v===Cn||v===gr?z=i.DEPTH_COMPONENT24:v===Yt?z=i.DEPTH_COMPONENT32F:v===mr&&(z=i.DEPTH_COMPONENT16),z}function U(C,v){return g(C)===!0||C.isFramebufferTexture&&C.minFilter!==Nt&&C.minFilter!==bt?Math.log2(Math.max(v.width,v.height))+1:C.mipmaps!==void 0&&C.mipmaps.length>0?C.mipmaps.length:C.isCompressedTexture&&Array.isArray(C.image)?v.mipmaps.length:1}function L(C){const v=C.target;v.removeEventListener("dispose",L),S(v),v.isVideoTexture&&u.delete(v)}function N(C){const v=C.target;v.removeEventListener("dispose",N),G(v)}function S(C){const v=n.get(C);if(v.__webglInit===void 0)return;const z=C.source,re=h.get(z);if(re){const le=re[v.__cacheKey];le.usedTimes--,le.usedTimes===0&&T(C),Object.keys(re).length===0&&h.delete(z)}n.remove(C)}function T(C){const v=n.get(C);i.deleteTexture(v.__webglTexture);const z=C.source,re=h.get(z);delete re[v.__cacheKey],a.memory.textures--}function G(C){const v=n.get(C);if(C.depthTexture&&(C.depthTexture.dispose(),n.remove(C.depthTexture)),C.isWebGLCubeRenderTarget)for(let re=0;re<6;re++){if(Array.isArray(v.__webglFramebuffer[re]))for(let le=0;le<v.__webglFramebuffer[re].length;le++)i.deleteFramebuffer(v.__webglFramebuffer[re][le]);else i.deleteFramebuffer(v.__webglFramebuffer[re]);v.__webglDepthbuffer&&i.deleteRenderbuffer(v.__webglDepthbuffer[re])}else{if(Array.isArray(v.__webglFramebuffer))for(let re=0;re<v.__webglFramebuffer.length;re++)i.deleteFramebuffer(v.__webglFramebuffer[re]);else i.deleteFramebuffer(v.__webglFramebuffer);if(v.__webglDepthbuffer&&i.deleteRenderbuffer(v.__webglDepthbuffer),v.__webglMultisampledFramebuffer&&i.deleteFramebuffer(v.__webglMultisampledFramebuffer),v.__webglColorRenderbuffer)for(let re=0;re<v.__webglColorRenderbuffer.length;re++)v.__webglColorRenderbuffer[re]&&i.deleteRenderbuffer(v.__webglColorRenderbuffer[re]);v.__webglDepthRenderbuffer&&i.deleteRenderbuffer(v.__webglDepthRenderbuffer)}const z=C.textures;for(let re=0,le=z.length;re<le;re++){const ne=n.get(z[re]);ne.__webglTexture&&(i.deleteTexture(ne.__webglTexture),a.memory.textures--),n.remove(z[re])}n.remove(C)}let D=0;function O(){D=0}function V(){const C=D;return C>=r.maxTextures&&Xe("WebGLTextures: Trying to use "+C+" texture units while this GPU supports only "+r.maxTextures),D+=1,C}function K(C){const v=[];return v.push(C.wrapS),v.push(C.wrapT),v.push(C.wrapR||0),v.push(C.magFilter),v.push(C.minFilter),v.push(C.anisotropy),v.push(C.internalFormat),v.push(C.format),v.push(C.type),v.push(C.generateMipmaps),v.push(C.premultiplyAlpha),v.push(C.flipY),v.push(C.unpackAlignment),v.push(C.colorSpace),v.join()}function Y(C,v){const z=n.get(C);if(C.isVideoTexture&&at(C),C.isRenderTargetTexture===!1&&C.isExternalTexture!==!0&&C.version>0&&z.__version!==C.version){const re=C.image;if(re===null)Xe("WebGLRenderer: Texture marked for update but no image data found.");else if(re.complete===!1)Xe("WebGLRenderer: Texture marked for update but image is incomplete");else{$(z,C,v);return}}else C.isExternalTexture&&(z.__webglTexture=C.sourceTexture?C.sourceTexture:null);t.bindTexture(i.TEXTURE_2D,z.__webglTexture,i.TEXTURE0+v)}function Z(C,v){const z=n.get(C);if(C.isRenderTargetTexture===!1&&C.version>0&&z.__version!==C.version){$(z,C,v);return}else C.isExternalTexture&&(z.__webglTexture=C.sourceTexture?C.sourceTexture:null);t.bindTexture(i.TEXTURE_2D_ARRAY,z.__webglTexture,i.TEXTURE0+v)}function X(C,v){const z=n.get(C);if(C.isRenderTargetTexture===!1&&C.version>0&&z.__version!==C.version){$(z,C,v);return}t.bindTexture(i.TEXTURE_3D,z.__webglTexture,i.TEXTURE0+v)}function fe(C,v){const z=n.get(C);if(C.isCubeDepthTexture!==!0&&C.version>0&&z.__version!==C.version){ue(z,C,v);return}t.bindTexture(i.TEXTURE_CUBE_MAP,z.__webglTexture,i.TEXTURE0+v)}const oe={[ga]:i.REPEAT,[gn]:i.CLAMP_TO_EDGE,[_a]:i.MIRRORED_REPEAT},ye={[Nt]:i.NEAREST,[Jc]:i.NEAREST_MIPMAP_NEAREST,[wr]:i.NEAREST_MIPMAP_LINEAR,[bt]:i.LINEAR,[Es]:i.LINEAR_MIPMAP_NEAREST,[ti]:i.LINEAR_MIPMAP_LINEAR},Ae={[th]:i.NEVER,[ah]:i.ALWAYS,[nh]:i.LESS,[uo]:i.LEQUAL,[ih]:i.EQUAL,[fo]:i.GEQUAL,[rh]:i.GREATER,[sh]:i.NOTEQUAL};function ve(C,v){if(v.type===Yt&&e.has("OES_texture_float_linear")===!1&&(v.magFilter===bt||v.magFilter===Es||v.magFilter===wr||v.magFilter===ti||v.minFilter===bt||v.minFilter===Es||v.minFilter===wr||v.minFilter===ti)&&Xe("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(C,i.TEXTURE_WRAP_S,oe[v.wrapS]),i.texParameteri(C,i.TEXTURE_WRAP_T,oe[v.wrapT]),(C===i.TEXTURE_3D||C===i.TEXTURE_2D_ARRAY)&&i.texParameteri(C,i.TEXTURE_WRAP_R,oe[v.wrapR]),i.texParameteri(C,i.TEXTURE_MAG_FILTER,ye[v.magFilter]),i.texParameteri(C,i.TEXTURE_MIN_FILTER,ye[v.minFilter]),v.compareFunction&&(i.texParameteri(C,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(C,i.TEXTURE_COMPARE_FUNC,Ae[v.compareFunction])),e.has("EXT_texture_filter_anisotropic")===!0){if(v.magFilter===Nt||v.minFilter!==wr&&v.minFilter!==ti||v.type===Yt&&e.has("OES_texture_float_linear")===!1)return;if(v.anisotropy>1||n.get(v).__currentAnisotropy){const z=e.get("EXT_texture_filter_anisotropic");i.texParameterf(C,z.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(v.anisotropy,r.getMaxAnisotropy())),n.get(v).__currentAnisotropy=v.anisotropy}}}function Ge(C,v){let z=!1;C.__webglInit===void 0&&(C.__webglInit=!0,v.addEventListener("dispose",L));const re=v.source;let le=h.get(re);le===void 0&&(le={},h.set(re,le));const ne=K(v);if(ne!==C.__cacheKey){le[ne]===void 0&&(le[ne]={texture:i.createTexture(),usedTimes:0},a.memory.textures++,z=!0),le[ne].usedTimes++;const Ce=le[C.__cacheKey];Ce!==void 0&&(le[C.__cacheKey].usedTimes--,Ce.usedTimes===0&&T(v)),C.__cacheKey=ne,C.__webglTexture=le[ne].texture}return z}function st(C,v,z){return Math.floor(Math.floor(C/z)/v)}function _e(C,v,z,re){const ne=C.updateRanges;if(ne.length===0)t.texSubImage2D(i.TEXTURE_2D,0,0,0,v.width,v.height,z,re,v.data);else{ne.sort((me,pe)=>me.start-pe.start);let Ce=0;for(let me=1;me<ne.length;me++){const pe=ne[Ce],Te=ne[me],Re=pe.start+pe.count,Ee=st(Te.start,v.width,4),He=st(pe.start,v.width,4);Te.start<=Re+1&&Ee===He&&st(Te.start+Te.count-1,v.width,4)===Ee?pe.count=Math.max(pe.count,Te.start+Te.count-pe.start):(++Ce,ne[Ce]=Te)}ne.length=Ce+1;const xe=i.getParameter(i.UNPACK_ROW_LENGTH),ke=i.getParameter(i.UNPACK_SKIP_PIXELS),Ve=i.getParameter(i.UNPACK_SKIP_ROWS);i.pixelStorei(i.UNPACK_ROW_LENGTH,v.width);for(let me=0,pe=ne.length;me<pe;me++){const Te=ne[me],Re=Math.floor(Te.start/4),Ee=Math.ceil(Te.count/4),He=Re%v.width,k=Math.floor(Re/v.width),Se=Ee,J=1;i.pixelStorei(i.UNPACK_SKIP_PIXELS,He),i.pixelStorei(i.UNPACK_SKIP_ROWS,k),t.texSubImage2D(i.TEXTURE_2D,0,He,k,Se,J,z,re,v.data)}C.clearUpdateRanges(),i.pixelStorei(i.UNPACK_ROW_LENGTH,xe),i.pixelStorei(i.UNPACK_SKIP_PIXELS,ke),i.pixelStorei(i.UNPACK_SKIP_ROWS,Ve)}}function $(C,v,z){let re=i.TEXTURE_2D;(v.isDataArrayTexture||v.isCompressedArrayTexture)&&(re=i.TEXTURE_2D_ARRAY),v.isData3DTexture&&(re=i.TEXTURE_3D);const le=Ge(C,v),ne=v.source;t.bindTexture(re,C.__webglTexture,i.TEXTURE0+z);const Ce=n.get(ne);if(ne.version!==Ce.__version||le===!0){t.activeTexture(i.TEXTURE0+z);const xe=lt.getPrimaries(lt.workingColorSpace),ke=v.colorSpace===ei?null:lt.getPrimaries(v.colorSpace),Ve=v.colorSpace===ei||xe===ke?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,v.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,v.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,Ve);let me=y(v.image,!1,r.maxTextureSize);me=pt(v,me);const pe=s.convert(v.format,v.colorSpace),Te=s.convert(v.type);let Re=w(v.internalFormat,pe,Te,v.colorSpace,v.isVideoTexture);ve(re,v);let Ee;const He=v.mipmaps,k=v.isVideoTexture!==!0,Se=Ce.__version===void 0||le===!0,J=ne.dataReady,we=U(v,me);if(v.isDepthTexture)Re=A(v.format===mi,v.type),Se&&(k?t.texStorage2D(i.TEXTURE_2D,1,Re,me.width,me.height):t.texImage2D(i.TEXTURE_2D,0,Re,me.width,me.height,0,pe,Te,null));else if(v.isDataTexture)if(He.length>0){k&&Se&&t.texStorage2D(i.TEXTURE_2D,we,Re,He[0].width,He[0].height);for(let ge=0,te=He.length;ge<te;ge++)Ee=He[ge],k?J&&t.texSubImage2D(i.TEXTURE_2D,ge,0,0,Ee.width,Ee.height,pe,Te,Ee.data):t.texImage2D(i.TEXTURE_2D,ge,Re,Ee.width,Ee.height,0,pe,Te,Ee.data);v.generateMipmaps=!1}else k?(Se&&t.texStorage2D(i.TEXTURE_2D,we,Re,me.width,me.height),J&&_e(v,me,pe,Te)):t.texImage2D(i.TEXTURE_2D,0,Re,me.width,me.height,0,pe,Te,me.data);else if(v.isCompressedTexture)if(v.isCompressedArrayTexture){k&&Se&&t.texStorage3D(i.TEXTURE_2D_ARRAY,we,Re,He[0].width,He[0].height,me.depth);for(let ge=0,te=He.length;ge<te;ge++)if(Ee=He[ge],v.format!==Ft)if(pe!==null)if(k){if(J)if(v.layerUpdates.size>0){const Pe=hl(Ee.width,Ee.height,v.format,v.type);for(const qe of v.layerUpdates){const mt=Ee.data.subarray(qe*Pe/Ee.data.BYTES_PER_ELEMENT,(qe+1)*Pe/Ee.data.BYTES_PER_ELEMENT);t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,ge,0,0,qe,Ee.width,Ee.height,1,pe,mt)}v.clearLayerUpdates()}else t.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,ge,0,0,0,Ee.width,Ee.height,me.depth,pe,Ee.data)}else t.compressedTexImage3D(i.TEXTURE_2D_ARRAY,ge,Re,Ee.width,Ee.height,me.depth,0,Ee.data,0,0);else Xe("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else k?J&&t.texSubImage3D(i.TEXTURE_2D_ARRAY,ge,0,0,0,Ee.width,Ee.height,me.depth,pe,Te,Ee.data):t.texImage3D(i.TEXTURE_2D_ARRAY,ge,Re,Ee.width,Ee.height,me.depth,0,pe,Te,Ee.data)}else{k&&Se&&t.texStorage2D(i.TEXTURE_2D,we,Re,He[0].width,He[0].height);for(let ge=0,te=He.length;ge<te;ge++)Ee=He[ge],v.format!==Ft?pe!==null?k?J&&t.compressedTexSubImage2D(i.TEXTURE_2D,ge,0,0,Ee.width,Ee.height,pe,Ee.data):t.compressedTexImage2D(i.TEXTURE_2D,ge,Re,Ee.width,Ee.height,0,Ee.data):Xe("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):k?J&&t.texSubImage2D(i.TEXTURE_2D,ge,0,0,Ee.width,Ee.height,pe,Te,Ee.data):t.texImage2D(i.TEXTURE_2D,ge,Re,Ee.width,Ee.height,0,pe,Te,Ee.data)}else if(v.isDataArrayTexture)if(k){if(Se&&t.texStorage3D(i.TEXTURE_2D_ARRAY,we,Re,me.width,me.height,me.depth),J)if(v.layerUpdates.size>0){const ge=hl(me.width,me.height,v.format,v.type);for(const te of v.layerUpdates){const Pe=me.data.subarray(te*ge/me.data.BYTES_PER_ELEMENT,(te+1)*ge/me.data.BYTES_PER_ELEMENT);t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,te,me.width,me.height,1,pe,Te,Pe)}v.clearLayerUpdates()}else t.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,me.width,me.height,me.depth,pe,Te,me.data)}else t.texImage3D(i.TEXTURE_2D_ARRAY,0,Re,me.width,me.height,me.depth,0,pe,Te,me.data);else if(v.isData3DTexture)k?(Se&&t.texStorage3D(i.TEXTURE_3D,we,Re,me.width,me.height,me.depth),J&&t.texSubImage3D(i.TEXTURE_3D,0,0,0,0,me.width,me.height,me.depth,pe,Te,me.data)):t.texImage3D(i.TEXTURE_3D,0,Re,me.width,me.height,me.depth,0,pe,Te,me.data);else if(v.isFramebufferTexture){if(Se)if(k)t.texStorage2D(i.TEXTURE_2D,we,Re,me.width,me.height);else{let ge=me.width,te=me.height;for(let Pe=0;Pe<we;Pe++)t.texImage2D(i.TEXTURE_2D,Pe,Re,ge,te,0,pe,Te,null),ge>>=1,te>>=1}}else if(He.length>0){if(k&&Se){const ge=De(He[0]);t.texStorage2D(i.TEXTURE_2D,we,Re,ge.width,ge.height)}for(let ge=0,te=He.length;ge<te;ge++)Ee=He[ge],k?J&&t.texSubImage2D(i.TEXTURE_2D,ge,0,0,pe,Te,Ee):t.texImage2D(i.TEXTURE_2D,ge,Re,pe,Te,Ee);v.generateMipmaps=!1}else if(k){if(Se){const ge=De(me);t.texStorage2D(i.TEXTURE_2D,we,Re,ge.width,ge.height)}J&&t.texSubImage2D(i.TEXTURE_2D,0,0,0,pe,Te,me)}else t.texImage2D(i.TEXTURE_2D,0,Re,pe,Te,me);g(v)&&m(re),Ce.__version=ne.version,v.onUpdate&&v.onUpdate(v)}C.__version=v.version}function ue(C,v,z){if(v.image.length!==6)return;const re=Ge(C,v),le=v.source;t.bindTexture(i.TEXTURE_CUBE_MAP,C.__webglTexture,i.TEXTURE0+z);const ne=n.get(le);if(le.version!==ne.__version||re===!0){t.activeTexture(i.TEXTURE0+z);const Ce=lt.getPrimaries(lt.workingColorSpace),xe=v.colorSpace===ei?null:lt.getPrimaries(v.colorSpace),ke=v.colorSpace===ei||Ce===xe?i.NONE:i.BROWSER_DEFAULT_WEBGL;i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,v.flipY),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,v.premultiplyAlpha),i.pixelStorei(i.UNPACK_ALIGNMENT,v.unpackAlignment),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,ke);const Ve=v.isCompressedTexture||v.image[0].isCompressedTexture,me=v.image[0]&&v.image[0].isDataTexture,pe=[];for(let te=0;te<6;te++)!Ve&&!me?pe[te]=y(v.image[te],!0,r.maxCubemapSize):pe[te]=me?v.image[te].image:v.image[te],pe[te]=pt(v,pe[te]);const Te=pe[0],Re=s.convert(v.format,v.colorSpace),Ee=s.convert(v.type),He=w(v.internalFormat,Re,Ee,v.colorSpace),k=v.isVideoTexture!==!0,Se=ne.__version===void 0||re===!0,J=le.dataReady;let we=U(v,Te);ve(i.TEXTURE_CUBE_MAP,v);let ge;if(Ve){k&&Se&&t.texStorage2D(i.TEXTURE_CUBE_MAP,we,He,Te.width,Te.height);for(let te=0;te<6;te++){ge=pe[te].mipmaps;for(let Pe=0;Pe<ge.length;Pe++){const qe=ge[Pe];v.format!==Ft?Re!==null?k?J&&t.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe,0,0,qe.width,qe.height,Re,qe.data):t.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe,He,qe.width,qe.height,0,qe.data):Xe("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):k?J&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe,0,0,qe.width,qe.height,Re,Ee,qe.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe,He,qe.width,qe.height,0,Re,Ee,qe.data)}}}else{if(ge=v.mipmaps,k&&Se){ge.length>0&&we++;const te=De(pe[0]);t.texStorage2D(i.TEXTURE_CUBE_MAP,we,He,te.width,te.height)}for(let te=0;te<6;te++)if(me){k?J&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,0,0,0,pe[te].width,pe[te].height,Re,Ee,pe[te].data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,0,He,pe[te].width,pe[te].height,0,Re,Ee,pe[te].data);for(let Pe=0;Pe<ge.length;Pe++){const mt=ge[Pe].image[te].image;k?J&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe+1,0,0,mt.width,mt.height,Re,Ee,mt.data):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe+1,He,mt.width,mt.height,0,Re,Ee,mt.data)}}else{k?J&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,0,0,0,Re,Ee,pe[te]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,0,He,Re,Ee,pe[te]);for(let Pe=0;Pe<ge.length;Pe++){const qe=ge[Pe];k?J&&t.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe+1,0,0,Re,Ee,qe.image[te]):t.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+te,Pe+1,He,Re,Ee,qe.image[te])}}}g(v)&&m(i.TEXTURE_CUBE_MAP),ne.__version=le.version,v.onUpdate&&v.onUpdate(v)}C.__version=v.version}function de(C,v,z,re,le,ne){const Ce=s.convert(z.format,z.colorSpace),xe=s.convert(z.type),ke=w(z.internalFormat,Ce,xe,z.colorSpace),Ve=n.get(v),me=n.get(z);if(me.__renderTarget=v,!Ve.__hasExternalTextures){const pe=Math.max(1,v.width>>ne),Te=Math.max(1,v.height>>ne);le===i.TEXTURE_3D||le===i.TEXTURE_2D_ARRAY?t.texImage3D(le,ne,ke,pe,Te,v.depth,0,Ce,xe,null):t.texImage2D(le,ne,ke,pe,Te,0,Ce,xe,null)}t.bindFramebuffer(i.FRAMEBUFFER,C),Mt(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,re,le,me.__webglTexture,0,B(v)):(le===i.TEXTURE_2D||le>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&le<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,re,le,me.__webglTexture,ne),t.bindFramebuffer(i.FRAMEBUFFER,null)}function ze(C,v,z){if(i.bindRenderbuffer(i.RENDERBUFFER,C),v.depthBuffer){const re=v.depthTexture,le=re&&re.isDepthTexture?re.type:null,ne=A(v.stencilBuffer,le),Ce=v.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;Mt(v)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,B(v),ne,v.width,v.height):z?i.renderbufferStorageMultisample(i.RENDERBUFFER,B(v),ne,v.width,v.height):i.renderbufferStorage(i.RENDERBUFFER,ne,v.width,v.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Ce,i.RENDERBUFFER,C)}else{const re=v.textures;for(let le=0;le<re.length;le++){const ne=re[le],Ce=s.convert(ne.format,ne.colorSpace),xe=s.convert(ne.type),ke=w(ne.internalFormat,Ce,xe,ne.colorSpace);Mt(v)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,B(v),ke,v.width,v.height):z?i.renderbufferStorageMultisample(i.RENDERBUFFER,B(v),ke,v.width,v.height):i.renderbufferStorage(i.RENDERBUFFER,ke,v.width,v.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function Le(C,v,z){const re=v.isWebGLCubeRenderTarget===!0;if(t.bindFramebuffer(i.FRAMEBUFFER,C),!(v.depthTexture&&v.depthTexture.isDepthTexture))throw new Error("renderTarget.depthTexture must be an instance of THREE.DepthTexture");const le=n.get(v.depthTexture);if(le.__renderTarget=v,(!le.__webglTexture||v.depthTexture.image.width!==v.width||v.depthTexture.image.height!==v.height)&&(v.depthTexture.image.width=v.width,v.depthTexture.image.height=v.height,v.depthTexture.needsUpdate=!0),re){if(le.__webglInit===void 0&&(le.__webglInit=!0,v.depthTexture.addEventListener("dispose",L)),le.__webglTexture===void 0){le.__webglTexture=i.createTexture(),t.bindTexture(i.TEXTURE_CUBE_MAP,le.__webglTexture),ve(i.TEXTURE_CUBE_MAP,v.depthTexture);const Ve=s.convert(v.depthTexture.format),me=s.convert(v.depthTexture.type);let pe;v.depthTexture.format===Hn?pe=i.DEPTH_COMPONENT24:v.depthTexture.format===mi&&(pe=i.DEPTH24_STENCIL8);for(let Te=0;Te<6;Te++)i.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Te,0,pe,v.width,v.height,0,Ve,me,null)}}else Y(v.depthTexture,0);const ne=le.__webglTexture,Ce=B(v),xe=re?i.TEXTURE_CUBE_MAP_POSITIVE_X+z:i.TEXTURE_2D,ke=v.depthTexture.format===mi?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;if(v.depthTexture.format===Hn)Mt(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,ke,xe,ne,0,Ce):i.framebufferTexture2D(i.FRAMEBUFFER,ke,xe,ne,0);else if(v.depthTexture.format===mi)Mt(v)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,ke,xe,ne,0,Ce):i.framebufferTexture2D(i.FRAMEBUFFER,ke,xe,ne,0);else throw new Error("Unknown depthTexture format")}function Fe(C){const v=n.get(C),z=C.isWebGLCubeRenderTarget===!0;if(v.__boundDepthTexture!==C.depthTexture){const re=C.depthTexture;if(v.__depthDisposeCallback&&v.__depthDisposeCallback(),re){const le=()=>{delete v.__boundDepthTexture,delete v.__depthDisposeCallback,re.removeEventListener("dispose",le)};re.addEventListener("dispose",le),v.__depthDisposeCallback=le}v.__boundDepthTexture=re}if(C.depthTexture&&!v.__autoAllocateDepthBuffer)if(z)for(let re=0;re<6;re++)Le(v.__webglFramebuffer[re],C,re);else{const re=C.texture.mipmaps;re&&re.length>0?Le(v.__webglFramebuffer[0],C,0):Le(v.__webglFramebuffer,C,0)}else if(z){v.__webglDepthbuffer=[];for(let re=0;re<6;re++)if(t.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer[re]),v.__webglDepthbuffer[re]===void 0)v.__webglDepthbuffer[re]=i.createRenderbuffer(),ze(v.__webglDepthbuffer[re],C,!1);else{const le=C.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ne=v.__webglDepthbuffer[re];i.bindRenderbuffer(i.RENDERBUFFER,ne),i.framebufferRenderbuffer(i.FRAMEBUFFER,le,i.RENDERBUFFER,ne)}}else{const re=C.texture.mipmaps;if(re&&re.length>0?t.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer[0]):t.bindFramebuffer(i.FRAMEBUFFER,v.__webglFramebuffer),v.__webglDepthbuffer===void 0)v.__webglDepthbuffer=i.createRenderbuffer(),ze(v.__webglDepthbuffer,C,!1);else{const le=C.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,ne=v.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,ne),i.framebufferRenderbuffer(i.FRAMEBUFFER,le,i.RENDERBUFFER,ne)}}t.bindFramebuffer(i.FRAMEBUFFER,null)}function xt(C,v,z){const re=n.get(C);v!==void 0&&de(re.__webglFramebuffer,C,C.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),z!==void 0&&Fe(C)}function et(C){const v=C.texture,z=n.get(C),re=n.get(v);C.addEventListener("dispose",N);const le=C.textures,ne=C.isWebGLCubeRenderTarget===!0,Ce=le.length>1;if(Ce||(re.__webglTexture===void 0&&(re.__webglTexture=i.createTexture()),re.__version=v.version,a.memory.textures++),ne){z.__webglFramebuffer=[];for(let xe=0;xe<6;xe++)if(v.mipmaps&&v.mipmaps.length>0){z.__webglFramebuffer[xe]=[];for(let ke=0;ke<v.mipmaps.length;ke++)z.__webglFramebuffer[xe][ke]=i.createFramebuffer()}else z.__webglFramebuffer[xe]=i.createFramebuffer()}else{if(v.mipmaps&&v.mipmaps.length>0){z.__webglFramebuffer=[];for(let xe=0;xe<v.mipmaps.length;xe++)z.__webglFramebuffer[xe]=i.createFramebuffer()}else z.__webglFramebuffer=i.createFramebuffer();if(Ce)for(let xe=0,ke=le.length;xe<ke;xe++){const Ve=n.get(le[xe]);Ve.__webglTexture===void 0&&(Ve.__webglTexture=i.createTexture(),a.memory.textures++)}if(C.samples>0&&Mt(C)===!1){z.__webglMultisampledFramebuffer=i.createFramebuffer(),z.__webglColorRenderbuffer=[],t.bindFramebuffer(i.FRAMEBUFFER,z.__webglMultisampledFramebuffer);for(let xe=0;xe<le.length;xe++){const ke=le[xe];z.__webglColorRenderbuffer[xe]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,z.__webglColorRenderbuffer[xe]);const Ve=s.convert(ke.format,ke.colorSpace),me=s.convert(ke.type),pe=w(ke.internalFormat,Ve,me,ke.colorSpace,C.isXRRenderTarget===!0),Te=B(C);i.renderbufferStorageMultisample(i.RENDERBUFFER,Te,pe,C.width,C.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+xe,i.RENDERBUFFER,z.__webglColorRenderbuffer[xe])}i.bindRenderbuffer(i.RENDERBUFFER,null),C.depthBuffer&&(z.__webglDepthRenderbuffer=i.createRenderbuffer(),ze(z.__webglDepthRenderbuffer,C,!0)),t.bindFramebuffer(i.FRAMEBUFFER,null)}}if(ne){t.bindTexture(i.TEXTURE_CUBE_MAP,re.__webglTexture),ve(i.TEXTURE_CUBE_MAP,v);for(let xe=0;xe<6;xe++)if(v.mipmaps&&v.mipmaps.length>0)for(let ke=0;ke<v.mipmaps.length;ke++)de(z.__webglFramebuffer[xe][ke],C,v,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+xe,ke);else de(z.__webglFramebuffer[xe],C,v,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+xe,0);g(v)&&m(i.TEXTURE_CUBE_MAP),t.unbindTexture()}else if(Ce){for(let xe=0,ke=le.length;xe<ke;xe++){const Ve=le[xe],me=n.get(Ve);let pe=i.TEXTURE_2D;(C.isWebGL3DRenderTarget||C.isWebGLArrayRenderTarget)&&(pe=C.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(pe,me.__webglTexture),ve(pe,Ve),de(z.__webglFramebuffer,C,Ve,i.COLOR_ATTACHMENT0+xe,pe,0),g(Ve)&&m(pe)}t.unbindTexture()}else{let xe=i.TEXTURE_2D;if((C.isWebGL3DRenderTarget||C.isWebGLArrayRenderTarget)&&(xe=C.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),t.bindTexture(xe,re.__webglTexture),ve(xe,v),v.mipmaps&&v.mipmaps.length>0)for(let ke=0;ke<v.mipmaps.length;ke++)de(z.__webglFramebuffer[ke],C,v,i.COLOR_ATTACHMENT0,xe,ke);else de(z.__webglFramebuffer,C,v,i.COLOR_ATTACHMENT0,xe,0);g(v)&&m(xe),t.unbindTexture()}C.depthBuffer&&Fe(C)}function ct(C){const v=C.textures;for(let z=0,re=v.length;z<re;z++){const le=v[z];if(g(le)){const ne=b(C),Ce=n.get(le).__webglTexture;t.bindTexture(ne,Ce),m(ne),t.unbindTexture()}}}const dt=[],Ke=[];function St(C){if(C.samples>0){if(Mt(C)===!1){const v=C.textures,z=C.width,re=C.height;let le=i.COLOR_BUFFER_BIT;const ne=C.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Ce=n.get(C),xe=v.length>1;if(xe)for(let Ve=0;Ve<v.length;Ve++)t.bindFramebuffer(i.FRAMEBUFFER,Ce.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ve,i.RENDERBUFFER,null),t.bindFramebuffer(i.FRAMEBUFFER,Ce.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ve,i.TEXTURE_2D,null,0);t.bindFramebuffer(i.READ_FRAMEBUFFER,Ce.__webglMultisampledFramebuffer);const ke=C.texture.mipmaps;ke&&ke.length>0?t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ce.__webglFramebuffer[0]):t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ce.__webglFramebuffer);for(let Ve=0;Ve<v.length;Ve++){if(C.resolveDepthBuffer&&(C.depthBuffer&&(le|=i.DEPTH_BUFFER_BIT),C.stencilBuffer&&C.resolveStencilBuffer&&(le|=i.STENCIL_BUFFER_BIT)),xe){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Ce.__webglColorRenderbuffer[Ve]);const me=n.get(v[Ve]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,me,0)}i.blitFramebuffer(0,0,z,re,0,0,z,re,le,i.NEAREST),c===!0&&(dt.length=0,Ke.length=0,dt.push(i.COLOR_ATTACHMENT0+Ve),C.depthBuffer&&C.resolveDepthBuffer===!1&&(dt.push(ne),Ke.push(ne),i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,Ke)),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,dt))}if(t.bindFramebuffer(i.READ_FRAMEBUFFER,null),t.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),xe)for(let Ve=0;Ve<v.length;Ve++){t.bindFramebuffer(i.FRAMEBUFFER,Ce.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ve,i.RENDERBUFFER,Ce.__webglColorRenderbuffer[Ve]);const me=n.get(v[Ve]).__webglTexture;t.bindFramebuffer(i.FRAMEBUFFER,Ce.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Ve,i.TEXTURE_2D,me,0)}t.bindFramebuffer(i.DRAW_FRAMEBUFFER,Ce.__webglMultisampledFramebuffer)}else if(C.depthBuffer&&C.resolveDepthBuffer===!1&&c){const v=C.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[v])}}}function B(C){return Math.min(r.maxSamples,C.samples)}function Mt(C){const v=n.get(C);return C.samples>0&&e.has("WEBGL_multisampled_render_to_texture")===!0&&v.__useRenderToTexture!==!1}function at(C){const v=a.render.frame;u.get(C)!==v&&(u.set(C,v),C.update())}function pt(C,v){const z=C.colorSpace,re=C.format,le=C.type;return C.isCompressedTexture===!0||C.isVideoTexture===!0||z!==jt&&z!==ei&&(lt.getTransfer(z)===ft?(re!==Ft||le!==Kt)&&Xe("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):ot("WebGLTextures: Unsupported texture color space:",z)),v}function De(C){return typeof HTMLImageElement<"u"&&C instanceof HTMLImageElement?(l.width=C.naturalWidth||C.width,l.height=C.naturalHeight||C.height):typeof VideoFrame<"u"&&C instanceof VideoFrame?(l.width=C.displayWidth,l.height=C.displayHeight):(l.width=C.width,l.height=C.height),l}this.allocateTextureUnit=V,this.resetTextureUnits=O,this.setTexture2D=Y,this.setTexture2DArray=Z,this.setTexture3D=X,this.setTextureCube=fe,this.rebindTextures=xt,this.setupRenderTarget=et,this.updateRenderTargetMipmap=ct,this.updateMultisampleRenderTarget=St,this.setupDepthRenderbuffer=Fe,this.setupFrameBufferTexture=de,this.useMultisampledRTT=Mt,this.isReversedDepthBuffer=function(){return t.buffers.depth.getReversed()}}function jm(i,e){function t(n,r=ei){let s;const a=lt.getTransfer(r);if(n===Kt)return i.UNSIGNED_BYTE;if(n===ao)return i.UNSIGNED_SHORT_4_4_4_4;if(n===oo)return i.UNSIGNED_SHORT_5_5_5_1;if(n===Kl)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===Jl)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===$l)return i.BYTE;if(n===jl)return i.SHORT;if(n===mr)return i.UNSIGNED_SHORT;if(n===so)return i.INT;if(n===Cn)return i.UNSIGNED_INT;if(n===Yt)return i.FLOAT;if(n===Jt)return i.HALF_FLOAT;if(n===Ql)return i.ALPHA;if(n===ec)return i.RGB;if(n===Ft)return i.RGBA;if(n===Hn)return i.DEPTH_COMPONENT;if(n===mi)return i.DEPTH_STENCIL;if(n===gi)return i.RED;if(n===lo)return i.RED_INTEGER;if(n===mn)return i.RG;if(n===co)return i.RG_INTEGER;if(n===ho)return i.RGBA_INTEGER;if(n===as||n===os||n===ls||n===cs)if(a===ft)if(s=e.get("WEBGL_compressed_texture_s3tc_srgb"),s!==null){if(n===as)return s.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===os)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===ls)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===cs)return s.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(s=e.get("WEBGL_compressed_texture_s3tc"),s!==null){if(n===as)return s.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===os)return s.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===ls)return s.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===cs)return s.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===xa||n===va||n===Sa||n===Ma)if(s=e.get("WEBGL_compressed_texture_pvrtc"),s!==null){if(n===xa)return s.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===va)return s.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===Sa)return s.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===Ma)return s.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===ya||n===Ea||n===ba||n===Ta||n===Aa||n===wa||n===Ca)if(s=e.get("WEBGL_compressed_texture_etc"),s!==null){if(n===ya||n===Ea)return a===ft?s.COMPRESSED_SRGB8_ETC2:s.COMPRESSED_RGB8_ETC2;if(n===ba)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:s.COMPRESSED_RGBA8_ETC2_EAC;if(n===Ta)return s.COMPRESSED_R11_EAC;if(n===Aa)return s.COMPRESSED_SIGNED_R11_EAC;if(n===wa)return s.COMPRESSED_RG11_EAC;if(n===Ca)return s.COMPRESSED_SIGNED_RG11_EAC}else return null;if(n===Ra||n===Pa||n===Da||n===Ia||n===La||n===Ua||n===Fa||n===Na||n===Oa||n===Ba||n===ka||n===za||n===Ga||n===Ha)if(s=e.get("WEBGL_compressed_texture_astc"),s!==null){if(n===Ra)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:s.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Pa)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:s.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===Da)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:s.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Ia)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:s.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===La)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:s.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===Ua)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:s.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===Fa)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:s.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===Na)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:s.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Oa)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:s.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Ba)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:s.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===ka)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:s.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===za)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:s.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===Ga)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:s.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===Ha)return a===ft?s.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:s.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===Va||n===Wa||n===Xa)if(s=e.get("EXT_texture_compression_bptc"),s!==null){if(n===Va)return a===ft?s.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:s.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===Wa)return s.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===Xa)return s.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Ya||n===qa||n===Za||n===$a)if(s=e.get("EXT_texture_compression_rgtc"),s!==null){if(n===Ya)return s.COMPRESSED_RED_RGTC1_EXT;if(n===qa)return s.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===Za)return s.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===$a)return s.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===gr?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:t}}const Km=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,Jm=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`;class Qm{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(e,t){if(this.texture===null){const n=new hc(e.texture);(e.depthNear!==t.depthNear||e.depthFar!==t.depthFar)&&(this.depthNear=e.depthNear,this.depthFar=e.depthFar),this.texture=n}}getMesh(e){if(this.texture!==null&&this.mesh===null){const t=e.cameras[0].viewport,n=new Pn({vertexShader:Km,fragmentShader:Jm,uniforms:{depthColor:{value:this.texture},depthWidth:{value:t.z},depthHeight:{value:t.w}}});this.mesh=new _n(new xs(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}}class eg extends xi{constructor(e,t){super();const n=this;let r=null,s=1,a=null,o="local-floor",c=1,l=null,u=null,d=null,h=null,f=null,_=null;const y=typeof XRWebGLBinding<"u",g=new Qm,m={},b=t.getContextAttributes();let w=null,A=null;const U=[],L=[],N=new $e;let S=null;const T=new rn;T.viewport=new Et;const G=new rn;G.viewport=new Et;const D=[T,G],O=new cu;let V=null,K=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function($){let ue=U[$];return ue===void 0&&(ue=new Ps,U[$]=ue),ue.getTargetRaySpace()},this.getControllerGrip=function($){let ue=U[$];return ue===void 0&&(ue=new Ps,U[$]=ue),ue.getGripSpace()},this.getHand=function($){let ue=U[$];return ue===void 0&&(ue=new Ps,U[$]=ue),ue.getHandSpace()};function Y($){const ue=L.indexOf($.inputSource);if(ue===-1)return;const de=U[ue];de!==void 0&&(de.update($.inputSource,$.frame,l||a),de.dispatchEvent({type:$.type,data:$.inputSource}))}function Z(){r.removeEventListener("select",Y),r.removeEventListener("selectstart",Y),r.removeEventListener("selectend",Y),r.removeEventListener("squeeze",Y),r.removeEventListener("squeezestart",Y),r.removeEventListener("squeezeend",Y),r.removeEventListener("end",Z),r.removeEventListener("inputsourceschange",X);for(let $=0;$<U.length;$++){const ue=L[$];ue!==null&&(L[$]=null,U[$].disconnect(ue))}V=null,K=null,g.reset();for(const $ in m)delete m[$];e.setRenderTarget(w),f=null,h=null,d=null,r=null,A=null,_e.stop(),n.isPresenting=!1,e.setPixelRatio(S),e.setSize(N.width,N.height,!1),n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function($){s=$,n.isPresenting===!0&&Xe("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function($){o=$,n.isPresenting===!0&&Xe("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||a},this.setReferenceSpace=function($){l=$},this.getBaseLayer=function(){return h!==null?h:f},this.getBinding=function(){return d===null&&y&&(d=new XRWebGLBinding(r,t)),d},this.getFrame=function(){return _},this.getSession=function(){return r},this.setSession=async function($){if(r=$,r!==null){if(w=e.getRenderTarget(),r.addEventListener("select",Y),r.addEventListener("selectstart",Y),r.addEventListener("selectend",Y),r.addEventListener("squeeze",Y),r.addEventListener("squeezestart",Y),r.addEventListener("squeezeend",Y),r.addEventListener("end",Z),r.addEventListener("inputsourceschange",X),b.xrCompatible!==!0&&await t.makeXRCompatible(),S=e.getPixelRatio(),e.getSize(N),y&&"createProjectionLayer"in XRWebGLBinding.prototype){let de=null,ze=null,Le=null;b.depth&&(Le=b.stencil?t.DEPTH24_STENCIL8:t.DEPTH_COMPONENT24,de=b.stencil?mi:Hn,ze=b.stencil?gr:Cn);const Fe={colorFormat:t.RGBA8,depthFormat:Le,scaleFactor:s};d=this.getBinding(),h=d.createProjectionLayer(Fe),r.updateRenderState({layers:[h]}),e.setPixelRatio(1),e.setSize(h.textureWidth,h.textureHeight,!1),A=new wn(h.textureWidth,h.textureHeight,{format:Ft,type:Kt,depthTexture:new xr(h.textureWidth,h.textureHeight,ze,void 0,void 0,void 0,void 0,void 0,void 0,de),stencilBuffer:b.stencil,colorSpace:e.outputColorSpace,samples:b.antialias?4:0,resolveDepthBuffer:h.ignoreDepthValues===!1,resolveStencilBuffer:h.ignoreDepthValues===!1})}else{const de={antialias:b.antialias,alpha:!0,depth:b.depth,stencil:b.stencil,framebufferScaleFactor:s};f=new XRWebGLLayer(r,t,de),r.updateRenderState({baseLayer:f}),e.setPixelRatio(1),e.setSize(f.framebufferWidth,f.framebufferHeight,!1),A=new wn(f.framebufferWidth,f.framebufferHeight,{format:Ft,type:Kt,colorSpace:e.outputColorSpace,stencilBuffer:b.stencil,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1})}A.isXRRenderTarget=!0,this.setFoveation(c),l=null,a=await r.requestReferenceSpace(o),_e.setContext(r),_e.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(r!==null)return r.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function X($){for(let ue=0;ue<$.removed.length;ue++){const de=$.removed[ue],ze=L.indexOf(de);ze>=0&&(L[ze]=null,U[ze].disconnect(de))}for(let ue=0;ue<$.added.length;ue++){const de=$.added[ue];let ze=L.indexOf(de);if(ze===-1){for(let Fe=0;Fe<U.length;Fe++)if(Fe>=L.length){L.push(de),ze=Fe;break}else if(L[Fe]===null){L[Fe]=de,ze=Fe;break}if(ze===-1)break}const Le=U[ze];Le&&Le.connect(de)}}const fe=new q,oe=new q;function ye($,ue,de){fe.setFromMatrixPosition(ue.matrixWorld),oe.setFromMatrixPosition(de.matrixWorld);const ze=fe.distanceTo(oe),Le=ue.projectionMatrix.elements,Fe=de.projectionMatrix.elements,xt=Le[14]/(Le[10]-1),et=Le[14]/(Le[10]+1),ct=(Le[9]+1)/Le[5],dt=(Le[9]-1)/Le[5],Ke=(Le[8]-1)/Le[0],St=(Fe[8]+1)/Fe[0],B=xt*Ke,Mt=xt*St,at=ze/(-Ke+St),pt=at*-Ke;if(ue.matrixWorld.decompose($.position,$.quaternion,$.scale),$.translateX(pt),$.translateZ(at),$.matrixWorld.compose($.position,$.quaternion,$.scale),$.matrixWorldInverse.copy($.matrixWorld).invert(),Le[10]===-1)$.projectionMatrix.copy(ue.projectionMatrix),$.projectionMatrixInverse.copy(ue.projectionMatrixInverse);else{const De=xt+at,C=et+at,v=B-pt,z=Mt+(ze-pt),re=ct*et/C*De,le=dt*et/C*De;$.projectionMatrix.makePerspective(v,z,re,le,De,C),$.projectionMatrixInverse.copy($.projectionMatrix).invert()}}function Ae($,ue){ue===null?$.matrixWorld.copy($.matrix):$.matrixWorld.multiplyMatrices(ue.matrixWorld,$.matrix),$.matrixWorldInverse.copy($.matrixWorld).invert()}this.updateCamera=function($){if(r===null)return;let ue=$.near,de=$.far;g.texture!==null&&(g.depthNear>0&&(ue=g.depthNear),g.depthFar>0&&(de=g.depthFar)),O.near=G.near=T.near=ue,O.far=G.far=T.far=de,(V!==O.near||K!==O.far)&&(r.updateRenderState({depthNear:O.near,depthFar:O.far}),V=O.near,K=O.far),O.layers.mask=$.layers.mask|6,T.layers.mask=O.layers.mask&-5,G.layers.mask=O.layers.mask&-3;const ze=$.parent,Le=O.cameras;Ae(O,ze);for(let Fe=0;Fe<Le.length;Fe++)Ae(Le[Fe],ze);Le.length===2?ye(O,T,G):O.projectionMatrix.copy(T.projectionMatrix),ve($,O,ze)};function ve($,ue,de){de===null?$.matrix.copy(ue.matrixWorld):($.matrix.copy(de.matrixWorld),$.matrix.invert(),$.matrix.multiply(ue.matrixWorld)),$.matrix.decompose($.position,$.quaternion,$.scale),$.updateMatrixWorld(!0),$.projectionMatrix.copy(ue.projectionMatrix),$.projectionMatrixInverse.copy(ue.projectionMatrixInverse),$.isPerspectiveCamera&&($.fov=ja*2*Math.atan(1/$.projectionMatrix.elements[5]),$.zoom=1)}this.getCamera=function(){return O},this.getFoveation=function(){if(!(h===null&&f===null))return c},this.setFoveation=function($){c=$,h!==null&&(h.fixedFoveation=$),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=$)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(O)},this.getCameraTexture=function($){return m[$]};let Ge=null;function st($,ue){if(u=ue.getViewerPose(l||a),_=ue,u!==null){const de=u.views;f!==null&&(e.setRenderTargetFramebuffer(A,f.framebuffer),e.setRenderTarget(A));let ze=!1;de.length!==O.cameras.length&&(O.cameras.length=0,ze=!0);for(let et=0;et<de.length;et++){const ct=de[et];let dt=null;if(f!==null)dt=f.getViewport(ct);else{const St=d.getViewSubImage(h,ct);dt=St.viewport,et===0&&(e.setRenderTargetTextures(A,St.colorTexture,St.depthStencilTexture),e.setRenderTarget(A))}let Ke=D[et];Ke===void 0&&(Ke=new rn,Ke.layers.enable(et),Ke.viewport=new Et,D[et]=Ke),Ke.matrix.fromArray(ct.transform.matrix),Ke.matrix.decompose(Ke.position,Ke.quaternion,Ke.scale),Ke.projectionMatrix.fromArray(ct.projectionMatrix),Ke.projectionMatrixInverse.copy(Ke.projectionMatrix).invert(),Ke.viewport.set(dt.x,dt.y,dt.width,dt.height),et===0&&(O.matrix.copy(Ke.matrix),O.matrix.decompose(O.position,O.quaternion,O.scale)),ze===!0&&O.cameras.push(Ke)}const Le=r.enabledFeatures;if(Le&&Le.includes("depth-sensing")&&r.depthUsage=="gpu-optimized"&&y){d=n.getBinding();const et=d.getDepthInformation(de[0]);et&&et.isValid&&et.texture&&g.init(et,r.renderState)}if(Le&&Le.includes("camera-access")&&y){e.state.unbindTexture(),d=n.getBinding();for(let et=0;et<de.length;et++){const ct=de[et].camera;if(ct){let dt=m[ct];dt||(dt=new hc,m[ct]=dt);const Ke=d.getCameraImage(ct);dt.sourceTexture=Ke}}}}for(let de=0;de<U.length;de++){const ze=L[de],Le=U[de];ze!==null&&Le!==void 0&&Le.update(ze,ue,l||a)}Ge&&Ge($,ue),ue.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:ue}),_=null}const _e=new pc;_e.setAnimationLoop(st),this.setAnimationLoop=function($){Ge=$},this.dispose=function(){}}}const di=new Rn,tg=new _t;function ng(i,e){function t(g,m){g.matrixAutoUpdate===!0&&g.updateMatrix(),m.value.copy(g.matrix)}function n(g,m){m.color.getRGB(g.fogColor.value,uc(i)),m.isFog?(g.fogNear.value=m.near,g.fogFar.value=m.far):m.isFogExp2&&(g.fogDensity.value=m.density)}function r(g,m,b,w,A){m.isMeshBasicMaterial?s(g,m):m.isMeshLambertMaterial?(s(g,m),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)):m.isMeshToonMaterial?(s(g,m),d(g,m)):m.isMeshPhongMaterial?(s(g,m),u(g,m),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)):m.isMeshStandardMaterial?(s(g,m),h(g,m),m.isMeshPhysicalMaterial&&f(g,m,A)):m.isMeshMatcapMaterial?(s(g,m),_(g,m)):m.isMeshDepthMaterial?s(g,m):m.isMeshDistanceMaterial?(s(g,m),y(g,m)):m.isMeshNormalMaterial?s(g,m):m.isLineBasicMaterial?(a(g,m),m.isLineDashedMaterial&&o(g,m)):m.isPointsMaterial?c(g,m,b,w):m.isSpriteMaterial?l(g,m):m.isShadowMaterial?(g.color.value.copy(m.color),g.opacity.value=m.opacity):m.isShaderMaterial&&(m.uniformsNeedUpdate=!1)}function s(g,m){g.opacity.value=m.opacity,m.color&&g.diffuse.value.copy(m.color),m.emissive&&g.emissive.value.copy(m.emissive).multiplyScalar(m.emissiveIntensity),m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.bumpMap&&(g.bumpMap.value=m.bumpMap,t(m.bumpMap,g.bumpMapTransform),g.bumpScale.value=m.bumpScale,m.side===qt&&(g.bumpScale.value*=-1)),m.normalMap&&(g.normalMap.value=m.normalMap,t(m.normalMap,g.normalMapTransform),g.normalScale.value.copy(m.normalScale),m.side===qt&&g.normalScale.value.negate()),m.displacementMap&&(g.displacementMap.value=m.displacementMap,t(m.displacementMap,g.displacementMapTransform),g.displacementScale.value=m.displacementScale,g.displacementBias.value=m.displacementBias),m.emissiveMap&&(g.emissiveMap.value=m.emissiveMap,t(m.emissiveMap,g.emissiveMapTransform)),m.specularMap&&(g.specularMap.value=m.specularMap,t(m.specularMap,g.specularMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest);const b=e.get(m),w=b.envMap,A=b.envMapRotation;w&&(g.envMap.value=w,di.copy(A),di.x*=-1,di.y*=-1,di.z*=-1,w.isCubeTexture&&w.isRenderTargetTexture===!1&&(di.y*=-1,di.z*=-1),g.envMapRotation.value.setFromMatrix4(tg.makeRotationFromEuler(di)),g.flipEnvMap.value=w.isCubeTexture&&w.isRenderTargetTexture===!1?-1:1,g.reflectivity.value=m.reflectivity,g.ior.value=m.ior,g.refractionRatio.value=m.refractionRatio),m.lightMap&&(g.lightMap.value=m.lightMap,g.lightMapIntensity.value=m.lightMapIntensity,t(m.lightMap,g.lightMapTransform)),m.aoMap&&(g.aoMap.value=m.aoMap,g.aoMapIntensity.value=m.aoMapIntensity,t(m.aoMap,g.aoMapTransform))}function a(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform))}function o(g,m){g.dashSize.value=m.dashSize,g.totalSize.value=m.dashSize+m.gapSize,g.scale.value=m.scale}function c(g,m,b,w){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.size.value=m.size*b,g.scale.value=w*.5,m.map&&(g.map.value=m.map,t(m.map,g.uvTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function l(g,m){g.diffuse.value.copy(m.color),g.opacity.value=m.opacity,g.rotation.value=m.rotation,m.map&&(g.map.value=m.map,t(m.map,g.mapTransform)),m.alphaMap&&(g.alphaMap.value=m.alphaMap,t(m.alphaMap,g.alphaMapTransform)),m.alphaTest>0&&(g.alphaTest.value=m.alphaTest)}function u(g,m){g.specular.value.copy(m.specular),g.shininess.value=Math.max(m.shininess,1e-4)}function d(g,m){m.gradientMap&&(g.gradientMap.value=m.gradientMap)}function h(g,m){g.metalness.value=m.metalness,m.metalnessMap&&(g.metalnessMap.value=m.metalnessMap,t(m.metalnessMap,g.metalnessMapTransform)),g.roughness.value=m.roughness,m.roughnessMap&&(g.roughnessMap.value=m.roughnessMap,t(m.roughnessMap,g.roughnessMapTransform)),m.envMap&&(g.envMapIntensity.value=m.envMapIntensity)}function f(g,m,b){g.ior.value=m.ior,m.sheen>0&&(g.sheenColor.value.copy(m.sheenColor).multiplyScalar(m.sheen),g.sheenRoughness.value=m.sheenRoughness,m.sheenColorMap&&(g.sheenColorMap.value=m.sheenColorMap,t(m.sheenColorMap,g.sheenColorMapTransform)),m.sheenRoughnessMap&&(g.sheenRoughnessMap.value=m.sheenRoughnessMap,t(m.sheenRoughnessMap,g.sheenRoughnessMapTransform))),m.clearcoat>0&&(g.clearcoat.value=m.clearcoat,g.clearcoatRoughness.value=m.clearcoatRoughness,m.clearcoatMap&&(g.clearcoatMap.value=m.clearcoatMap,t(m.clearcoatMap,g.clearcoatMapTransform)),m.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=m.clearcoatRoughnessMap,t(m.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),m.clearcoatNormalMap&&(g.clearcoatNormalMap.value=m.clearcoatNormalMap,t(m.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(m.clearcoatNormalScale),m.side===qt&&g.clearcoatNormalScale.value.negate())),m.dispersion>0&&(g.dispersion.value=m.dispersion),m.iridescence>0&&(g.iridescence.value=m.iridescence,g.iridescenceIOR.value=m.iridescenceIOR,g.iridescenceThicknessMinimum.value=m.iridescenceThicknessRange[0],g.iridescenceThicknessMaximum.value=m.iridescenceThicknessRange[1],m.iridescenceMap&&(g.iridescenceMap.value=m.iridescenceMap,t(m.iridescenceMap,g.iridescenceMapTransform)),m.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=m.iridescenceThicknessMap,t(m.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),m.transmission>0&&(g.transmission.value=m.transmission,g.transmissionSamplerMap.value=b.texture,g.transmissionSamplerSize.value.set(b.width,b.height),m.transmissionMap&&(g.transmissionMap.value=m.transmissionMap,t(m.transmissionMap,g.transmissionMapTransform)),g.thickness.value=m.thickness,m.thicknessMap&&(g.thicknessMap.value=m.thicknessMap,t(m.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=m.attenuationDistance,g.attenuationColor.value.copy(m.attenuationColor)),m.anisotropy>0&&(g.anisotropyVector.value.set(m.anisotropy*Math.cos(m.anisotropyRotation),m.anisotropy*Math.sin(m.anisotropyRotation)),m.anisotropyMap&&(g.anisotropyMap.value=m.anisotropyMap,t(m.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=m.specularIntensity,g.specularColor.value.copy(m.specularColor),m.specularColorMap&&(g.specularColorMap.value=m.specularColorMap,t(m.specularColorMap,g.specularColorMapTransform)),m.specularIntensityMap&&(g.specularIntensityMap.value=m.specularIntensityMap,t(m.specularIntensityMap,g.specularIntensityMapTransform))}function _(g,m){m.matcap&&(g.matcap.value=m.matcap)}function y(g,m){const b=e.get(m).light;g.referencePosition.value.setFromMatrixPosition(b.matrixWorld),g.nearDistance.value=b.shadow.camera.near,g.farDistance.value=b.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:r}}function ig(i,e,t,n){let r={},s={},a=[];const o=i.getParameter(i.MAX_UNIFORM_BUFFER_BINDINGS);function c(b,w){const A=w.program;n.uniformBlockBinding(b,A)}function l(b,w){let A=r[b.id];A===void 0&&(_(b),A=u(b),r[b.id]=A,b.addEventListener("dispose",g));const U=w.program;n.updateUBOMapping(b,U);const L=e.render.frame;s[b.id]!==L&&(h(b),s[b.id]=L)}function u(b){const w=d();b.__bindingPointIndex=w;const A=i.createBuffer(),U=b.__size,L=b.usage;return i.bindBuffer(i.UNIFORM_BUFFER,A),i.bufferData(i.UNIFORM_BUFFER,U,L),i.bindBuffer(i.UNIFORM_BUFFER,null),i.bindBufferBase(i.UNIFORM_BUFFER,w,A),A}function d(){for(let b=0;b<o;b++)if(a.indexOf(b)===-1)return a.push(b),b;return ot("WebGLRenderer: Maximum number of simultaneously usable uniforms groups reached."),0}function h(b){const w=r[b.id],A=b.uniforms,U=b.__cache;i.bindBuffer(i.UNIFORM_BUFFER,w);for(let L=0,N=A.length;L<N;L++){const S=Array.isArray(A[L])?A[L]:[A[L]];for(let T=0,G=S.length;T<G;T++){const D=S[T];if(f(D,L,T,U)===!0){const O=D.__offset,V=Array.isArray(D.value)?D.value:[D.value];let K=0;for(let Y=0;Y<V.length;Y++){const Z=V[Y],X=y(Z);typeof Z=="number"||typeof Z=="boolean"?(D.__data[0]=Z,i.bufferSubData(i.UNIFORM_BUFFER,O+K,D.__data)):Z.isMatrix3?(D.__data[0]=Z.elements[0],D.__data[1]=Z.elements[1],D.__data[2]=Z.elements[2],D.__data[3]=0,D.__data[4]=Z.elements[3],D.__data[5]=Z.elements[4],D.__data[6]=Z.elements[5],D.__data[7]=0,D.__data[8]=Z.elements[6],D.__data[9]=Z.elements[7],D.__data[10]=Z.elements[8],D.__data[11]=0):(Z.toArray(D.__data,K),K+=X.storage/Float32Array.BYTES_PER_ELEMENT)}i.bufferSubData(i.UNIFORM_BUFFER,O,D.__data)}}}i.bindBuffer(i.UNIFORM_BUFFER,null)}function f(b,w,A,U){const L=b.value,N=w+"_"+A;if(U[N]===void 0)return typeof L=="number"||typeof L=="boolean"?U[N]=L:U[N]=L.clone(),!0;{const S=U[N];if(typeof L=="number"||typeof L=="boolean"){if(S!==L)return U[N]=L,!0}else if(S.equals(L)===!1)return S.copy(L),!0}return!1}function _(b){const w=b.uniforms;let A=0;const U=16;for(let N=0,S=w.length;N<S;N++){const T=Array.isArray(w[N])?w[N]:[w[N]];for(let G=0,D=T.length;G<D;G++){const O=T[G],V=Array.isArray(O.value)?O.value:[O.value];for(let K=0,Y=V.length;K<Y;K++){const Z=V[K],X=y(Z),fe=A%U,oe=fe%X.boundary,ye=fe+oe;A+=oe,ye!==0&&U-ye<X.storage&&(A+=U-ye),O.__data=new Float32Array(X.storage/Float32Array.BYTES_PER_ELEMENT),O.__offset=A,A+=X.storage}}}const L=A%U;return L>0&&(A+=U-L),b.__size=A,b.__cache={},this}function y(b){const w={boundary:0,storage:0};return typeof b=="number"||typeof b=="boolean"?(w.boundary=4,w.storage=4):b.isVector2?(w.boundary=8,w.storage=8):b.isVector3||b.isColor?(w.boundary=16,w.storage=12):b.isVector4?(w.boundary=16,w.storage=16):b.isMatrix3?(w.boundary=48,w.storage=48):b.isMatrix4?(w.boundary=64,w.storage=64):b.isTexture?Xe("WebGLRenderer: Texture samplers can not be part of an uniforms group."):Xe("WebGLRenderer: Unsupported uniform value type.",b),w}function g(b){const w=b.target;w.removeEventListener("dispose",g);const A=a.indexOf(w.__bindingPointIndex);a.splice(A,1),i.deleteBuffer(r[w.id]),delete r[w.id],delete s[w.id]}function m(){for(const b in r)i.deleteBuffer(r[b]);a=[],r={},s={}}return{bind:c,update:l,dispose:m}}const rg=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]);let yn=null;function sg(){return yn===null&&(yn=new go(rg,16,16,mn,Jt),yn.name="DFG_LUT",yn.minFilter=bt,yn.magFilter=bt,yn.wrapS=gn,yn.wrapT=gn,yn.generateMipmaps=!1,yn.needsUpdate=!0),yn}class ag{constructor(e={}){const{canvas:t=ch(),context:n=null,depth:r=!0,stencil:s=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:d=!1,reversedDepthBuffer:h=!1,outputBufferType:f=Kt}=e;this.isWebGLRenderer=!0;let _;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");_=n.getContextAttributes().alpha}else _=a;const y=f,g=new Set([ho,co,lo]),m=new Set([Kt,Cn,mr,gr,ao,oo]),b=new Uint32Array(4),w=new Int32Array(4);let A=null,U=null;const L=[],N=[];let S=null;this.domElement=t,this.debug={checkShaderErrors:!0,onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=An,this.toneMappingExposure=1,this.transmissionResolutionScale=1;const T=this;let G=!1;this._outputColorSpace=nn;let D=0,O=0,V=null,K=-1,Y=null;const Z=new Et,X=new Et;let fe=null;const oe=new rt(0);let ye=0,Ae=t.width,ve=t.height,Ge=1,st=null,_e=null;const $=new Et(0,0,Ae,ve),ue=new Et(0,0,Ae,ve);let de=!1;const ze=new _o;let Le=!1,Fe=!1;const xt=new _t,et=new q,ct=new Et,dt={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0};let Ke=!1;function St(){return V===null?Ge:1}let B=n;function Mt(x,M){return t.getContext(x,M)}try{const x={alpha:!0,depth:r,stencil:s,antialias:o,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:u,failIfMajorPerformanceCaveat:d};if("setAttribute"in t&&t.setAttribute("data-engine",`three.js r${io}`),t.addEventListener("webglcontextlost",Pe,!1),t.addEventListener("webglcontextrestored",qe,!1),t.addEventListener("webglcontextcreationerror",mt,!1),B===null){const M="webgl2";if(B=Mt(M,x),B===null)throw Mt(M)?new Error("Error creating WebGL context with your selected attributes."):new Error("Error creating WebGL context.")}}catch(x){throw ot("WebGLRenderer: "+x.message),x}let at,pt,De,C,v,z,re,le,ne,Ce,xe,ke,Ve,me,pe,Te,Re,Ee,He,k,Se,J,we;function ge(){at=new ap(B),at.init(),Se=new jm(B,at),pt=new Jf(B,at,e,Se),De=new Zm(B,at),pt.reversedDepthBuffer&&h&&De.buffers.depth.setReversed(!0),C=new cp(B),v=new Um,z=new $m(B,at,De,v,pt,Se,C),re=new sp(T),le=new fu(B),J=new jf(B,le),ne=new op(B,le,C,J),Ce=new up(B,ne,le,J,C),Ee=new hp(B,pt,z),pe=new Qf(v),xe=new Lm(T,re,at,pt,J,pe),ke=new ng(T,v),Ve=new Nm,me=new Hm(at),Re=new $f(T,re,De,Ce,_,c),Te=new qm(T,Ce,pt),we=new ig(B,C,pt,De),He=new Kf(B,at,C),k=new lp(B,at,C),C.programs=xe.programs,T.capabilities=pt,T.extensions=at,T.properties=v,T.renderLists=Ve,T.shadowMap=Te,T.state=De,T.info=C}ge(),y!==Kt&&(S=new fp(y,t.width,t.height,r,s));const te=new eg(T,B);this.xr=te,this.getContext=function(){return B},this.getContextAttributes=function(){return B.getContextAttributes()},this.forceContextLoss=function(){const x=at.get("WEBGL_lose_context");x&&x.loseContext()},this.forceContextRestore=function(){const x=at.get("WEBGL_lose_context");x&&x.restoreContext()},this.getPixelRatio=function(){return Ge},this.setPixelRatio=function(x){x!==void 0&&(Ge=x,this.setSize(Ae,ve,!1))},this.getSize=function(x){return x.set(Ae,ve)},this.setSize=function(x,M,F=!0){if(te.isPresenting){Xe("WebGLRenderer: Can't change size while VR device is presenting.");return}Ae=x,ve=M,t.width=Math.floor(x*Ge),t.height=Math.floor(M*Ge),F===!0&&(t.style.width=x+"px",t.style.height=M+"px"),S!==null&&S.setSize(t.width,t.height),this.setViewport(0,0,x,M)},this.getDrawingBufferSize=function(x){return x.set(Ae*Ge,ve*Ge).floor()},this.setDrawingBufferSize=function(x,M,F){Ae=x,ve=M,Ge=F,t.width=Math.floor(x*F),t.height=Math.floor(M*F),this.setViewport(0,0,x,M)},this.setEffects=function(x){if(y===Kt){console.error("THREE.WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(x){for(let M=0;M<x.length;M++)if(x[M].isOutputPass===!0){console.warn("THREE.WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}S.setEffects(x||[])},this.getCurrentViewport=function(x){return x.copy(Z)},this.getViewport=function(x){return x.copy($)},this.setViewport=function(x,M,F,P){x.isVector4?$.set(x.x,x.y,x.z,x.w):$.set(x,M,F,P),De.viewport(Z.copy($).multiplyScalar(Ge).round())},this.getScissor=function(x){return x.copy(ue)},this.setScissor=function(x,M,F,P){x.isVector4?ue.set(x.x,x.y,x.z,x.w):ue.set(x,M,F,P),De.scissor(X.copy(ue).multiplyScalar(Ge).round())},this.getScissorTest=function(){return de},this.setScissorTest=function(x){De.setScissorTest(de=x)},this.setOpaqueSort=function(x){st=x},this.setTransparentSort=function(x){_e=x},this.getClearColor=function(x){return x.copy(Re.getClearColor())},this.setClearColor=function(){Re.setClearColor(...arguments)},this.getClearAlpha=function(){return Re.getClearAlpha()},this.setClearAlpha=function(){Re.setClearAlpha(...arguments)},this.clear=function(x=!0,M=!0,F=!0){let P=0;if(x){let R=!1;if(V!==null){const W=V.texture.format;R=g.has(W)}if(R){const W=V.texture.type,Q=m.has(W),j=Re.getClearColor(),ee=Re.getClearAlpha(),ae=j.r,se=j.g,he=j.b;Q?(b[0]=ae,b[1]=se,b[2]=he,b[3]=ee,B.clearBufferuiv(B.COLOR,0,b)):(w[0]=ae,w[1]=se,w[2]=he,w[3]=ee,B.clearBufferiv(B.COLOR,0,w))}else P|=B.COLOR_BUFFER_BIT}M&&(P|=B.DEPTH_BUFFER_BIT),F&&(P|=B.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),P!==0&&B.clear(P)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.dispose=function(){t.removeEventListener("webglcontextlost",Pe,!1),t.removeEventListener("webglcontextrestored",qe,!1),t.removeEventListener("webglcontextcreationerror",mt,!1),Re.dispose(),Ve.dispose(),me.dispose(),v.dispose(),re.dispose(),Ce.dispose(),J.dispose(),we.dispose(),xe.dispose(),te.dispose(),te.removeEventListener("sessionstart",yr),te.removeEventListener("sessionend",Er),xn.stop()};function Pe(x){x.preventDefault(),Io("WebGLRenderer: Context Lost."),G=!0}function qe(){Io("WebGLRenderer: Context Restored."),G=!1;const x=C.autoReset,M=Te.enabled,F=Te.autoUpdate,P=Te.needsUpdate,R=Te.type;ge(),C.autoReset=x,Te.enabled=M,Te.autoUpdate=F,Te.needsUpdate=P,Te.type=R}function mt(x){ot("WebGLRenderer: A WebGL context could not be created. Reason: ",x.statusMessage)}function ut(x){const M=x.target;M.removeEventListener("dispose",ut),ln(M)}function ln(x){cn(x),v.remove(x)}function cn(x){const M=v.get(x).programs;M!==void 0&&(M.forEach(function(F){xe.releaseProgram(F)}),x.isShaderMaterial&&xe.releaseShaderCache(x))}this.renderBufferDirect=function(x,M,F,P,R,W){M===null&&(M=dt);const Q=R.isMesh&&R.matrixWorld.determinant()<0,j=Sn(x,M,F,P,R);De.setMaterial(P,Q);let ee=F.index,ae=1;if(P.wireframe===!0){if(ee=ne.getWireframeAttribute(F),ee===void 0)return;ae=2}const se=F.drawRange,he=F.attributes.position;let ce=se.start*ae,Ie=(se.start+se.count)*ae;W!==null&&(ce=Math.max(ce,W.start*ae),Ie=Math.min(Ie,(W.start+W.count)*ae)),ee!==null?(ce=Math.max(ce,0),Ie=Math.min(Ie,ee.count)):he!=null&&(ce=Math.max(ce,0),Ie=Math.min(Ie,he.count));const We=Ie-ce;if(We<0||We===1/0)return;J.setup(R,P,j,F,ee);let je,Oe=He;if(ee!==null&&(je=le.get(ee),Oe=k,Oe.setIndex(je)),R.isMesh)P.wireframe===!0?(De.setLineWidth(P.wireframeLinewidth*St()),Oe.setMode(B.LINES)):Oe.setMode(B.TRIANGLES);else if(R.isLine){let Ue=P.linewidth;Ue===void 0&&(Ue=1),De.setLineWidth(Ue*St()),R.isLineSegments?Oe.setMode(B.LINES):R.isLineLoop?Oe.setMode(B.LINE_LOOP):Oe.setMode(B.LINE_STRIP)}else R.isPoints?Oe.setMode(B.POINTS):R.isSprite&&Oe.setMode(B.TRIANGLES);if(R.isBatchedMesh)if(R._multiDrawInstances!==null)fs("WebGLRenderer: renderMultiDrawInstances has been deprecated and will be removed in r184. Append to renderMultiDraw arguments and use indirection."),Oe.renderMultiDrawInstances(R._multiDrawStarts,R._multiDrawCounts,R._multiDrawCount,R._multiDrawInstances);else if(at.get("WEBGL_multi_draw"))Oe.renderMultiDraw(R._multiDrawStarts,R._multiDrawCounts,R._multiDrawCount);else{const Ue=R._multiDrawStarts,Me=R._multiDrawCounts,ht=R._multiDrawCount,Ne=ee?le.get(ee).bytesPerElement:1,Ze=v.get(P).currentProgram.getUniforms();for(let it=0;it<ht;it++)Ze.setValue(B,"_gl_DrawID",it),Oe.render(Ue[it]/Ne,Me[it])}else if(R.isInstancedMesh)Oe.renderInstances(ce,We,R.count);else if(F.isInstancedBufferGeometry){const Ue=F._maxInstanceCount!==void 0?F._maxInstanceCount:1/0,Me=Math.min(F.instanceCount,Ue);Oe.renderInstances(ce,We,Me)}else Oe.render(ce,We)};function Mr(x,M,F){x.transparent===!0&&x.side===Bn&&x.forceSinglePass===!1?(x.side=qt,x.needsUpdate=!0,ai(x,M,F),x.side=ri,x.needsUpdate=!0,ai(x,M,F),x.side=Bn):ai(x,M,F)}this.compile=function(x,M,F=null){F===null&&(F=x),U=me.get(F),U.init(M),N.push(U),F.traverseVisible(function(R){R.isLight&&R.layers.test(M.layers)&&(U.pushLight(R),R.castShadow&&U.pushShadow(R))}),x!==F&&x.traverseVisible(function(R){R.isLight&&R.layers.test(M.layers)&&(U.pushLight(R),R.castShadow&&U.pushShadow(R))}),U.setupLights();const P=new Set;return x.traverse(function(R){if(!(R.isMesh||R.isPoints||R.isLine||R.isSprite))return;const W=R.material;if(W)if(Array.isArray(W))for(let Q=0;Q<W.length;Q++){const j=W[Q];Mr(j,F,R),P.add(j)}else Mr(W,F,R),P.add(W)}),U=N.pop(),P},this.compileAsync=function(x,M,F=null){const P=this.compile(x,M,F);return new Promise(R=>{function W(){if(P.forEach(function(Q){v.get(Q).currentProgram.isReady()&&P.delete(Q)}),P.size===0){R(x);return}setTimeout(W,10)}at.get("KHR_parallel_shader_compile")!==null?W():setTimeout(W,10)})};let $i=null;function Ms(x){$i&&$i(x)}function yr(){xn.stop()}function Er(){xn.start()}const xn=new pc;xn.setAnimationLoop(Ms),typeof self<"u"&&xn.setContext(self),this.setAnimationLoop=function(x){$i=x,te.setAnimationLoop(x),x===null?xn.stop():xn.start()},te.addEventListener("sessionstart",yr),te.addEventListener("sessionend",Er),this.render=function(x,M){if(M!==void 0&&M.isCamera!==!0){ot("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(G===!0)return;const F=te.enabled===!0&&te.isPresenting===!0,P=S!==null&&(V===null||F)&&S.begin(T,V);if(x.matrixWorldAutoUpdate===!0&&x.updateMatrixWorld(),M.parent===null&&M.matrixWorldAutoUpdate===!0&&M.updateMatrixWorld(),te.enabled===!0&&te.isPresenting===!0&&(S===null||S.isCompositing()===!1)&&(te.cameraAutoUpdate===!0&&te.updateCamera(M),M=te.getCamera()),x.isScene===!0&&x.onBeforeRender(T,x,M,V),U=me.get(x,N.length),U.init(M),N.push(U),xt.multiplyMatrices(M.projectionMatrix,M.matrixWorldInverse),ze.setFromProjectionMatrix(xt,Tn,M.reversedDepth),Fe=this.localClippingEnabled,Le=pe.init(this.clippingPlanes,Fe),A=Ve.get(x,L.length),A.init(),L.push(A),te.enabled===!0&&te.isPresenting===!0){const Q=T.xr.getDepthSensingMesh();Q!==null&&ji(Q,M,-1/0,T.sortObjects)}ji(x,M,0,T.sortObjects),A.finish(),T.sortObjects===!0&&A.sort(st,_e),Ke=te.enabled===!1||te.isPresenting===!1||te.hasDepthSensing()===!1,Ke&&Re.addToRenderList(A,x),this.info.render.frame++,Le===!0&&pe.beginShadows();const R=U.state.shadowsArray;if(Te.render(R,x,M),Le===!0&&pe.endShadows(),this.info.autoReset===!0&&this.info.reset(),(P&&S.hasRenderPass())===!1){const Q=A.opaque,j=A.transmissive;if(U.setupLights(),M.isArrayCamera){const ee=M.cameras;if(j.length>0)for(let ae=0,se=ee.length;ae<se;ae++){const he=ee[ae];Tr(Q,j,x,he)}Ke&&Re.render(x);for(let ae=0,se=ee.length;ae<se;ae++){const he=ee[ae];br(A,x,he,he.viewport)}}else j.length>0&&Tr(Q,j,x,M),Ke&&Re.render(x),br(A,x,M)}V!==null&&O===0&&(z.updateMultisampleRenderTarget(V),z.updateRenderTargetMipmap(V)),P&&S.end(T),x.isScene===!0&&x.onAfterRender(T,x,M),J.resetDefaultState(),K=-1,Y=null,N.pop(),N.length>0?(U=N[N.length-1],Le===!0&&pe.setGlobalState(T.clippingPlanes,U.state.camera)):U=null,L.pop(),L.length>0?A=L[L.length-1]:A=null};function ji(x,M,F,P){if(x.visible===!1)return;if(x.layers.test(M.layers)){if(x.isGroup)F=x.renderOrder;else if(x.isLOD)x.autoUpdate===!0&&x.update(M);else if(x.isLight)U.pushLight(x),x.castShadow&&U.pushShadow(x);else if(x.isSprite){if(!x.frustumCulled||ze.intersectsSprite(x)){P&&ct.setFromMatrixPosition(x.matrixWorld).applyMatrix4(xt);const Q=Ce.update(x),j=x.material;j.visible&&A.push(x,Q,j,F,ct.z,null)}}else if((x.isMesh||x.isLine||x.isPoints)&&(!x.frustumCulled||ze.intersectsObject(x))){const Q=Ce.update(x),j=x.material;if(P&&(x.boundingSphere!==void 0?(x.boundingSphere===null&&x.computeBoundingSphere(),ct.copy(x.boundingSphere.center)):(Q.boundingSphere===null&&Q.computeBoundingSphere(),ct.copy(Q.boundingSphere.center)),ct.applyMatrix4(x.matrixWorld).applyMatrix4(xt)),Array.isArray(j)){const ee=Q.groups;for(let ae=0,se=ee.length;ae<se;ae++){const he=ee[ae],ce=j[he.materialIndex];ce&&ce.visible&&A.push(x,Q,ce,F,ct.z,he)}}else j.visible&&A.push(x,Q,j,F,ct.z,null)}}const W=x.children;for(let Q=0,j=W.length;Q<j;Q++)ji(W[Q],M,F,P)}function br(x,M,F,P){const{opaque:R,transmissive:W,transparent:Q}=x;U.setupLightsView(F),Le===!0&&pe.setGlobalState(T.clippingPlanes,F),P&&De.viewport(Z.copy(P)),R.length>0&&Si(R,M,F),W.length>0&&Si(W,M,F),Q.length>0&&Si(Q,M,F),De.buffers.depth.setTest(!0),De.buffers.depth.setMask(!0),De.buffers.color.setMask(!0),De.setPolygonOffset(!1)}function Tr(x,M,F,P){if((F.isScene===!0?F.overrideMaterial:null)!==null)return;if(U.state.transmissionRenderTarget[P.id]===void 0){const ce=at.has("EXT_color_buffer_half_float")||at.has("EXT_color_buffer_float");U.state.transmissionRenderTarget[P.id]=new wn(1,1,{generateMipmaps:!0,type:ce?Jt:Kt,minFilter:ti,samples:pt.samples,stencilBuffer:s,resolveDepthBuffer:!1,resolveStencilBuffer:!1,colorSpace:lt.workingColorSpace})}const W=U.state.transmissionRenderTarget[P.id],Q=P.viewport||Z;W.setSize(Q.z*T.transmissionResolutionScale,Q.w*T.transmissionResolutionScale);const j=T.getRenderTarget(),ee=T.getActiveCubeFace(),ae=T.getActiveMipmapLevel();T.setRenderTarget(W),T.getClearColor(oe),ye=T.getClearAlpha(),ye<1&&T.setClearColor(16777215,.5),T.clear(),Ke&&Re.render(F);const se=T.toneMapping;T.toneMapping=An;const he=P.viewport;if(P.viewport!==void 0&&(P.viewport=void 0),U.setupLightsView(P),Le===!0&&pe.setGlobalState(T.clippingPlanes,P),Si(x,F,P),z.updateMultisampleRenderTarget(W),z.updateRenderTargetMipmap(W),at.has("WEBGL_multisampled_render_to_texture")===!1){let ce=!1;for(let Ie=0,We=M.length;Ie<We;Ie++){const je=M[Ie],{object:Oe,geometry:Ue,material:Me,group:ht}=je;if(Me.side===Bn&&Oe.layers.test(P.layers)){const Ne=Me.side;Me.side=qt,Me.needsUpdate=!0,Ki(Oe,F,P,Ue,Me,ht),Me.side=Ne,Me.needsUpdate=!0,ce=!0}}ce===!0&&(z.updateMultisampleRenderTarget(W),z.updateRenderTargetMipmap(W))}T.setRenderTarget(j,ee,ae),T.setClearColor(oe,ye),he!==void 0&&(P.viewport=he),T.toneMapping=se}function Si(x,M,F){const P=M.isScene===!0?M.overrideMaterial:null;for(let R=0,W=x.length;R<W;R++){const Q=x[R],{object:j,geometry:ee,group:ae}=Q;let se=Q.material;se.allowOverride===!0&&P!==null&&(se=P),j.layers.test(F.layers)&&Ki(j,M,F,ee,se,ae)}}function Ki(x,M,F,P,R,W){x.onBeforeRender(T,M,F,P,R,W),x.modelViewMatrix.multiplyMatrices(F.matrixWorldInverse,x.matrixWorld),x.normalMatrix.getNormalMatrix(x.modelViewMatrix),R.onBeforeRender(T,M,F,P,x,W),R.transparent===!0&&R.side===Bn&&R.forceSinglePass===!1?(R.side=qt,R.needsUpdate=!0,T.renderBufferDirect(F,M,P,R,x,W),R.side=ri,R.needsUpdate=!0,T.renderBufferDirect(F,M,P,R,x,W),R.side=Bn):T.renderBufferDirect(F,M,P,R,x,W),x.onAfterRender(T,M,F,P,R,W)}function ai(x,M,F){M.isScene!==!0&&(M=dt);const P=v.get(x),R=U.state.lights,W=U.state.shadowsArray,Q=R.state.version,j=xe.getParameters(x,R.state,W,M,F),ee=xe.getProgramCacheKey(j);let ae=P.programs;P.environment=x.isMeshStandardMaterial||x.isMeshLambertMaterial||x.isMeshPhongMaterial?M.environment:null,P.fog=M.fog;const se=x.isMeshStandardMaterial||x.isMeshLambertMaterial&&!x.envMap||x.isMeshPhongMaterial&&!x.envMap;P.envMap=re.get(x.envMap||P.environment,se),P.envMapRotation=P.environment!==null&&x.envMap===null?M.environmentRotation:x.envMapRotation,ae===void 0&&(x.addEventListener("dispose",ut),ae=new Map,P.programs=ae);let he=ae.get(ee);if(he!==void 0){if(P.currentProgram===he&&P.lightsStateVersion===Q)return vn(x,j),he}else j.uniforms=xe.getUniforms(x),x.onBeforeCompile(j,T),he=xe.acquireProgram(j,ee),ae.set(ee,he),P.uniforms=j.uniforms;const ce=P.uniforms;return(!x.isShaderMaterial&&!x.isRawShaderMaterial||x.clipping===!0)&&(ce.clippingPlanes=pe.uniform),vn(x,j),P.needsLights=E(x),P.lightsStateVersion=Q,P.needsLights&&(ce.ambientLightColor.value=R.state.ambient,ce.lightProbe.value=R.state.probe,ce.directionalLights.value=R.state.directional,ce.directionalLightShadows.value=R.state.directionalShadow,ce.spotLights.value=R.state.spot,ce.spotLightShadows.value=R.state.spotShadow,ce.rectAreaLights.value=R.state.rectArea,ce.ltc_1.value=R.state.rectAreaLTC1,ce.ltc_2.value=R.state.rectAreaLTC2,ce.pointLights.value=R.state.point,ce.pointLightShadows.value=R.state.pointShadow,ce.hemisphereLights.value=R.state.hemi,ce.directionalShadowMatrix.value=R.state.directionalShadowMatrix,ce.spotLightMatrix.value=R.state.spotLightMatrix,ce.spotLightMap.value=R.state.spotLightMap,ce.pointShadowMatrix.value=R.state.pointShadowMatrix),P.currentProgram=he,P.uniformsList=null,he}function Ar(x){if(x.uniformsList===null){const M=x.currentProgram.getUniforms();x.uniformsList=hs.seqWithValue(M.seq,x.uniforms)}return x.uniformsList}function vn(x,M){const F=v.get(x);F.outputColorSpace=M.outputColorSpace,F.batching=M.batching,F.batchingColor=M.batchingColor,F.instancing=M.instancing,F.instancingColor=M.instancingColor,F.instancingMorph=M.instancingMorph,F.skinning=M.skinning,F.morphTargets=M.morphTargets,F.morphNormals=M.morphNormals,F.morphColors=M.morphColors,F.morphTargetsCount=M.morphTargetsCount,F.numClippingPlanes=M.numClippingPlanes,F.numIntersection=M.numClipIntersection,F.vertexAlphas=M.vertexAlphas,F.vertexTangents=M.vertexTangents,F.toneMapping=M.toneMapping}function Sn(x,M,F,P,R){M.isScene!==!0&&(M=dt),z.resetTextureUnits();const W=M.fog,Q=P.isMeshStandardMaterial||P.isMeshLambertMaterial||P.isMeshPhongMaterial?M.environment:null,j=V===null?T.outputColorSpace:V.isXRRenderTarget===!0?V.texture.colorSpace:jt,ee=P.isMeshStandardMaterial||P.isMeshLambertMaterial&&!P.envMap||P.isMeshPhongMaterial&&!P.envMap,ae=re.get(P.envMap||Q,ee),se=P.vertexColors===!0&&!!F.attributes.color&&F.attributes.color.itemSize===4,he=!!F.attributes.tangent&&(!!P.normalMap||P.anisotropy>0),ce=!!F.morphAttributes.position,Ie=!!F.morphAttributes.normal,We=!!F.morphAttributes.color;let je=An;P.toneMapped&&(V===null||V.isXRRenderTarget===!0)&&(je=T.toneMapping);const Oe=F.morphAttributes.position||F.morphAttributes.normal||F.morphAttributes.color,Ue=Oe!==void 0?Oe.length:0,Me=v.get(P),ht=U.state.lights;if(Le===!0&&(Fe===!0||x!==Y)){const Pt=x===Y&&P.id===K;pe.setState(P,x,Pt)}let Ne=!1;P.version===Me.__version?(Me.needsLights&&Me.lightsStateVersion!==ht.state.version||Me.outputColorSpace!==j||R.isBatchedMesh&&Me.batching===!1||!R.isBatchedMesh&&Me.batching===!0||R.isBatchedMesh&&Me.batchingColor===!0&&R.colorTexture===null||R.isBatchedMesh&&Me.batchingColor===!1&&R.colorTexture!==null||R.isInstancedMesh&&Me.instancing===!1||!R.isInstancedMesh&&Me.instancing===!0||R.isSkinnedMesh&&Me.skinning===!1||!R.isSkinnedMesh&&Me.skinning===!0||R.isInstancedMesh&&Me.instancingColor===!0&&R.instanceColor===null||R.isInstancedMesh&&Me.instancingColor===!1&&R.instanceColor!==null||R.isInstancedMesh&&Me.instancingMorph===!0&&R.morphTexture===null||R.isInstancedMesh&&Me.instancingMorph===!1&&R.morphTexture!==null||Me.envMap!==ae||P.fog===!0&&Me.fog!==W||Me.numClippingPlanes!==void 0&&(Me.numClippingPlanes!==pe.numPlanes||Me.numIntersection!==pe.numIntersection)||Me.vertexAlphas!==se||Me.vertexTangents!==he||Me.morphTargets!==ce||Me.morphNormals!==Ie||Me.morphColors!==We||Me.toneMapping!==je||Me.morphTargetsCount!==Ue)&&(Ne=!0):(Ne=!0,Me.__version=P.version);let Ze=Me.currentProgram;Ne===!0&&(Ze=ai(P,M,R));let it=!1,Lt=!1,en=!1;const tt=Ze.getUniforms(),yt=Me.uniforms;if(De.useProgram(Ze.program)&&(it=!0,Lt=!0,en=!0),P.id!==K&&(K=P.id,Lt=!0),it||Y!==x){De.buffers.depth.getReversed()&&x.reversedDepth!==!0&&(x._reversedDepth=!0,x.updateProjectionMatrix()),tt.setValue(B,"projectionMatrix",x.projectionMatrix),tt.setValue(B,"viewMatrix",x.matrixWorldInverse);const Wn=tt.map.cameraPosition;Wn!==void 0&&Wn.setValue(B,et.setFromMatrixPosition(x.matrixWorld)),pt.logarithmicDepthBuffer&&tt.setValue(B,"logDepthBufFC",2/(Math.log(x.far+1)/Math.LN2)),(P.isMeshPhongMaterial||P.isMeshToonMaterial||P.isMeshLambertMaterial||P.isMeshBasicMaterial||P.isMeshStandardMaterial||P.isShaderMaterial)&&tt.setValue(B,"isOrthographic",x.isOrthographicCamera===!0),Y!==x&&(Y=x,Lt=!0,en=!0)}if(Me.needsLights&&(ht.state.directionalShadowMap.length>0&&tt.setValue(B,"directionalShadowMap",ht.state.directionalShadowMap,z),ht.state.spotShadowMap.length>0&&tt.setValue(B,"spotShadowMap",ht.state.spotShadowMap,z),ht.state.pointShadowMap.length>0&&tt.setValue(B,"pointShadowMap",ht.state.pointShadowMap,z)),R.isSkinnedMesh){tt.setOptional(B,R,"bindMatrix"),tt.setOptional(B,R,"bindMatrixInverse");const Pt=R.skeleton;Pt&&(Pt.boneTexture===null&&Pt.computeBoneTexture(),tt.setValue(B,"boneTexture",Pt.boneTexture,z))}R.isBatchedMesh&&(tt.setOptional(B,R,"batchingTexture"),tt.setValue(B,"batchingTexture",R._matricesTexture,z),tt.setOptional(B,R,"batchingIdTexture"),tt.setValue(B,"batchingIdTexture",R._indirectTexture,z),tt.setOptional(B,R,"batchingColorTexture"),R._colorsTexture!==null&&tt.setValue(B,"batchingColorTexture",R._colorsTexture,z));const Vn=F.morphAttributes;if((Vn.position!==void 0||Vn.normal!==void 0||Vn.color!==void 0)&&Ee.update(R,F,Ze),(Lt||Me.receiveShadow!==R.receiveShadow)&&(Me.receiveShadow=R.receiveShadow,tt.setValue(B,"receiveShadow",R.receiveShadow)),(P.isMeshStandardMaterial||P.isMeshLambertMaterial||P.isMeshPhongMaterial)&&P.envMap===null&&M.environment!==null&&(yt.envMapIntensity.value=M.environmentIntensity),yt.dfgLUT!==void 0&&(yt.dfgLUT.value=sg()),Lt&&(tt.setValue(B,"toneMappingExposure",T.toneMappingExposure),Me.needsLights&&p(yt,en),W&&P.fog===!0&&ke.refreshFogUniforms(yt,W),ke.refreshMaterialUniforms(yt,P,Ge,ve,U.state.transmissionRenderTarget[x.id]),hs.upload(B,Ar(Me),yt,z)),P.isShaderMaterial&&P.uniformsNeedUpdate===!0&&(hs.upload(B,Ar(Me),yt,z),P.uniformsNeedUpdate=!1),P.isSpriteMaterial&&tt.setValue(B,"center",R.center),tt.setValue(B,"modelViewMatrix",R.modelViewMatrix),tt.setValue(B,"normalMatrix",R.normalMatrix),tt.setValue(B,"modelMatrix",R.matrixWorld),P.isShaderMaterial||P.isRawShaderMaterial){const Pt=P.uniformsGroups;for(let Wn=0,Mi=Pt.length;Wn<Mi;Wn++){const Eo=Pt[Wn];we.update(Eo,Ze),we.bind(Eo,Ze)}}return Ze}function p(x,M){x.ambientLightColor.needsUpdate=M,x.lightProbe.needsUpdate=M,x.directionalLights.needsUpdate=M,x.directionalLightShadows.needsUpdate=M,x.pointLights.needsUpdate=M,x.pointLightShadows.needsUpdate=M,x.spotLights.needsUpdate=M,x.spotLightShadows.needsUpdate=M,x.rectAreaLights.needsUpdate=M,x.hemisphereLights.needsUpdate=M}function E(x){return x.isMeshLambertMaterial||x.isMeshToonMaterial||x.isMeshPhongMaterial||x.isMeshStandardMaterial||x.isShadowMaterial||x.isShaderMaterial&&x.lights===!0}this.getActiveCubeFace=function(){return D},this.getActiveMipmapLevel=function(){return O},this.getRenderTarget=function(){return V},this.setRenderTargetTextures=function(x,M,F){const P=v.get(x);P.__autoAllocateDepthBuffer=x.resolveDepthBuffer===!1,P.__autoAllocateDepthBuffer===!1&&(P.__useRenderToTexture=!1),v.get(x.texture).__webglTexture=M,v.get(x.depthTexture).__webglTexture=P.__autoAllocateDepthBuffer?void 0:F,P.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(x,M){const F=v.get(x);F.__webglFramebuffer=M,F.__useDefaultFramebuffer=M===void 0};const I=B.createFramebuffer();this.setRenderTarget=function(x,M=0,F=0){V=x,D=M,O=F;let P=null,R=!1,W=!1;if(x){const j=v.get(x);if(j.__useDefaultFramebuffer!==void 0){De.bindFramebuffer(B.FRAMEBUFFER,j.__webglFramebuffer),Z.copy(x.viewport),X.copy(x.scissor),fe=x.scissorTest,De.viewport(Z),De.scissor(X),De.setScissorTest(fe),K=-1;return}else if(j.__webglFramebuffer===void 0)z.setupRenderTarget(x);else if(j.__hasExternalTextures)z.rebindTextures(x,v.get(x.texture).__webglTexture,v.get(x.depthTexture).__webglTexture);else if(x.depthBuffer){const se=x.depthTexture;if(j.__boundDepthTexture!==se){if(se!==null&&v.has(se)&&(x.width!==se.image.width||x.height!==se.image.height))throw new Error("WebGLRenderTarget: Attached DepthTexture is initialized to the incorrect size.");z.setupDepthRenderbuffer(x)}}const ee=x.texture;(ee.isData3DTexture||ee.isDataArrayTexture||ee.isCompressedArrayTexture)&&(W=!0);const ae=v.get(x).__webglFramebuffer;x.isWebGLCubeRenderTarget?(Array.isArray(ae[M])?P=ae[M][F]:P=ae[M],R=!0):x.samples>0&&z.useMultisampledRTT(x)===!1?P=v.get(x).__webglMultisampledFramebuffer:Array.isArray(ae)?P=ae[F]:P=ae,Z.copy(x.viewport),X.copy(x.scissor),fe=x.scissorTest}else Z.copy($).multiplyScalar(Ge).floor(),X.copy(ue).multiplyScalar(Ge).floor(),fe=de;if(F!==0&&(P=I),De.bindFramebuffer(B.FRAMEBUFFER,P)&&De.drawBuffers(x,P),De.viewport(Z),De.scissor(X),De.setScissorTest(fe),R){const j=v.get(x.texture);B.framebufferTexture2D(B.FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_CUBE_MAP_POSITIVE_X+M,j.__webglTexture,F)}else if(W){const j=M;for(let ee=0;ee<x.textures.length;ee++){const ae=v.get(x.textures[ee]);B.framebufferTextureLayer(B.FRAMEBUFFER,B.COLOR_ATTACHMENT0+ee,ae.__webglTexture,F,j)}}else if(x!==null&&F!==0){const j=v.get(x.texture);B.framebufferTexture2D(B.FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_2D,j.__webglTexture,F)}K=-1},this.readRenderTargetPixels=function(x,M,F,P,R,W,Q,j=0){if(!(x&&x.isWebGLRenderTarget)){ot("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let ee=v.get(x).__webglFramebuffer;if(x.isWebGLCubeRenderTarget&&Q!==void 0&&(ee=ee[Q]),ee){De.bindFramebuffer(B.FRAMEBUFFER,ee);try{const ae=x.textures[j],se=ae.format,he=ae.type;if(x.textures.length>1&&B.readBuffer(B.COLOR_ATTACHMENT0+j),!pt.textureFormatReadable(se)){ot("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(!pt.textureTypeReadable(he)){ot("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}M>=0&&M<=x.width-P&&F>=0&&F<=x.height-R&&B.readPixels(M,F,P,R,Se.convert(se),Se.convert(he),W)}finally{const ae=V!==null?v.get(V).__webglFramebuffer:null;De.bindFramebuffer(B.FRAMEBUFFER,ae)}}},this.readRenderTargetPixelsAsync=async function(x,M,F,P,R,W,Q,j=0){if(!(x&&x.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let ee=v.get(x).__webglFramebuffer;if(x.isWebGLCubeRenderTarget&&Q!==void 0&&(ee=ee[Q]),ee)if(M>=0&&M<=x.width-P&&F>=0&&F<=x.height-R){De.bindFramebuffer(B.FRAMEBUFFER,ee);const ae=x.textures[j],se=ae.format,he=ae.type;if(x.textures.length>1&&B.readBuffer(B.COLOR_ATTACHMENT0+j),!pt.textureFormatReadable(se))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(!pt.textureTypeReadable(he))throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");const ce=B.createBuffer();B.bindBuffer(B.PIXEL_PACK_BUFFER,ce),B.bufferData(B.PIXEL_PACK_BUFFER,W.byteLength,B.STREAM_READ),B.readPixels(M,F,P,R,Se.convert(se),Se.convert(he),0);const Ie=V!==null?v.get(V).__webglFramebuffer:null;De.bindFramebuffer(B.FRAMEBUFFER,Ie);const We=B.fenceSync(B.SYNC_GPU_COMMANDS_COMPLETE,0);return B.flush(),await hh(B,We,4),B.bindBuffer(B.PIXEL_PACK_BUFFER,ce),B.getBufferSubData(B.PIXEL_PACK_BUFFER,0,W),B.deleteBuffer(ce),B.deleteSync(We),W}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(x,M=null,F=0){const P=Math.pow(2,-F),R=Math.floor(x.image.width*P),W=Math.floor(x.image.height*P),Q=M!==null?M.x:0,j=M!==null?M.y:0;z.setTexture2D(x,0),B.copyTexSubImage2D(B.TEXTURE_2D,F,0,0,Q,j,R,W),De.unbindTexture()};const H=B.createFramebuffer(),ie=B.createFramebuffer();this.copyTextureToTexture=function(x,M,F=null,P=null,R=0,W=0){let Q,j,ee,ae,se,he,ce,Ie,We;const je=x.isCompressedTexture?x.mipmaps[W]:x.image;if(F!==null)Q=F.max.x-F.min.x,j=F.max.y-F.min.y,ee=F.isBox3?F.max.z-F.min.z:1,ae=F.min.x,se=F.min.y,he=F.isBox3?F.min.z:0;else{const yt=Math.pow(2,-R);Q=Math.floor(je.width*yt),j=Math.floor(je.height*yt),x.isDataArrayTexture?ee=je.depth:x.isData3DTexture?ee=Math.floor(je.depth*yt):ee=1,ae=0,se=0,he=0}P!==null?(ce=P.x,Ie=P.y,We=P.z):(ce=0,Ie=0,We=0);const Oe=Se.convert(M.format),Ue=Se.convert(M.type);let Me;M.isData3DTexture?(z.setTexture3D(M,0),Me=B.TEXTURE_3D):M.isDataArrayTexture||M.isCompressedArrayTexture?(z.setTexture2DArray(M,0),Me=B.TEXTURE_2D_ARRAY):(z.setTexture2D(M,0),Me=B.TEXTURE_2D),B.pixelStorei(B.UNPACK_FLIP_Y_WEBGL,M.flipY),B.pixelStorei(B.UNPACK_PREMULTIPLY_ALPHA_WEBGL,M.premultiplyAlpha),B.pixelStorei(B.UNPACK_ALIGNMENT,M.unpackAlignment);const ht=B.getParameter(B.UNPACK_ROW_LENGTH),Ne=B.getParameter(B.UNPACK_IMAGE_HEIGHT),Ze=B.getParameter(B.UNPACK_SKIP_PIXELS),it=B.getParameter(B.UNPACK_SKIP_ROWS),Lt=B.getParameter(B.UNPACK_SKIP_IMAGES);B.pixelStorei(B.UNPACK_ROW_LENGTH,je.width),B.pixelStorei(B.UNPACK_IMAGE_HEIGHT,je.height),B.pixelStorei(B.UNPACK_SKIP_PIXELS,ae),B.pixelStorei(B.UNPACK_SKIP_ROWS,se),B.pixelStorei(B.UNPACK_SKIP_IMAGES,he);const en=x.isDataArrayTexture||x.isData3DTexture,tt=M.isDataArrayTexture||M.isData3DTexture;if(x.isDepthTexture){const yt=v.get(x),Vn=v.get(M),Pt=v.get(yt.__renderTarget),Wn=v.get(Vn.__renderTarget);De.bindFramebuffer(B.READ_FRAMEBUFFER,Pt.__webglFramebuffer),De.bindFramebuffer(B.DRAW_FRAMEBUFFER,Wn.__webglFramebuffer);for(let Mi=0;Mi<ee;Mi++)en&&(B.framebufferTextureLayer(B.READ_FRAMEBUFFER,B.COLOR_ATTACHMENT0,v.get(x).__webglTexture,R,he+Mi),B.framebufferTextureLayer(B.DRAW_FRAMEBUFFER,B.COLOR_ATTACHMENT0,v.get(M).__webglTexture,W,We+Mi)),B.blitFramebuffer(ae,se,Q,j,ce,Ie,Q,j,B.DEPTH_BUFFER_BIT,B.NEAREST);De.bindFramebuffer(B.READ_FRAMEBUFFER,null),De.bindFramebuffer(B.DRAW_FRAMEBUFFER,null)}else if(R!==0||x.isRenderTargetTexture||v.has(x)){const yt=v.get(x),Vn=v.get(M);De.bindFramebuffer(B.READ_FRAMEBUFFER,H),De.bindFramebuffer(B.DRAW_FRAMEBUFFER,ie);for(let Pt=0;Pt<ee;Pt++)en?B.framebufferTextureLayer(B.READ_FRAMEBUFFER,B.COLOR_ATTACHMENT0,yt.__webglTexture,R,he+Pt):B.framebufferTexture2D(B.READ_FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_2D,yt.__webglTexture,R),tt?B.framebufferTextureLayer(B.DRAW_FRAMEBUFFER,B.COLOR_ATTACHMENT0,Vn.__webglTexture,W,We+Pt):B.framebufferTexture2D(B.DRAW_FRAMEBUFFER,B.COLOR_ATTACHMENT0,B.TEXTURE_2D,Vn.__webglTexture,W),R!==0?B.blitFramebuffer(ae,se,Q,j,ce,Ie,Q,j,B.COLOR_BUFFER_BIT,B.NEAREST):tt?B.copyTexSubImage3D(Me,W,ce,Ie,We+Pt,ae,se,Q,j):B.copyTexSubImage2D(Me,W,ce,Ie,ae,se,Q,j);De.bindFramebuffer(B.READ_FRAMEBUFFER,null),De.bindFramebuffer(B.DRAW_FRAMEBUFFER,null)}else tt?x.isDataTexture||x.isData3DTexture?B.texSubImage3D(Me,W,ce,Ie,We,Q,j,ee,Oe,Ue,je.data):M.isCompressedArrayTexture?B.compressedTexSubImage3D(Me,W,ce,Ie,We,Q,j,ee,Oe,je.data):B.texSubImage3D(Me,W,ce,Ie,We,Q,j,ee,Oe,Ue,je):x.isDataTexture?B.texSubImage2D(B.TEXTURE_2D,W,ce,Ie,Q,j,Oe,Ue,je.data):x.isCompressedTexture?B.compressedTexSubImage2D(B.TEXTURE_2D,W,ce,Ie,je.width,je.height,Oe,je.data):B.texSubImage2D(B.TEXTURE_2D,W,ce,Ie,Q,j,Oe,Ue,je);B.pixelStorei(B.UNPACK_ROW_LENGTH,ht),B.pixelStorei(B.UNPACK_IMAGE_HEIGHT,Ne),B.pixelStorei(B.UNPACK_SKIP_PIXELS,Ze),B.pixelStorei(B.UNPACK_SKIP_ROWS,it),B.pixelStorei(B.UNPACK_SKIP_IMAGES,Lt),W===0&&M.generateMipmaps&&B.generateMipmap(Me),De.unbindTexture()},this.initRenderTarget=function(x){v.get(x).__webglFramebuffer===void 0&&z.setupRenderTarget(x)},this.initTexture=function(x){x.isCubeTexture?z.setTextureCube(x,0):x.isData3DTexture?z.setTexture3D(x,0):x.isDataArrayTexture||x.isCompressedArrayTexture?z.setTexture2DArray(x,0):z.setTexture2D(x,0),De.unbindTexture()},this.resetState=function(){D=0,O=0,V=null,De.reset(),J.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return Tn}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(e){this._outputColorSpace=e;const t=this.getContext();t.drawingBufferColorSpace=lt._getDrawingBufferColorSpace(e),t.unpackColorSpace=lt._getUnpackColorSpace()}}const Ul=256,Oi={x:800,y:800,z:800},og=60,lg=500,Be={SEPARATION_DIST:0,ALIGN_DIST:1,COHESION_DIST:2,MAX_SPEED:3,MAX_FORCE:4,SEPARATION_WEIGHT:5,ALIGNMENT_WEIGHT:6,COHESION_WEIGHT:7,MARGIN:8,TURN_FACTOR:9,CELL_SIZE:10,PADDING:11,WORLD_MAX:12,GRID_DIM:16,MOUSE_RAY_ORIGIN:20,VISION_ANGLE:23,RAY_DIRECTION:24,FLEE_RADIUS:27};function cg(i){return{min:{x:0,y:0,z:0},max:{x:i.x,y:i.y,z:i.z}}}function Fl(i,e){const t=Oi.x*Oi.y*Oi.z,n=i/e,r=Math.cbrt(n/t);return{x:Oi.x*r,y:Oi.y*r,z:Oi.z*r}}function hg(i,e){const t=Math.min(i[Be.SEPARATION_DIST],i[Be.ALIGN_DIST],i[Be.COHESION_DIST],50),n={x:Math.max(1,Math.ceil(e.x/t)),y:Math.max(1,Math.ceil(e.y/t)),z:Math.max(1,Math.ceil(e.z/t))},r=n.x*n.y*n.z;return{cellSize:t,gridDim:n,numCells:r}}class ug{constructor(){this.gpuDevice=null,this.paramsArray=new Float32Array(28),this.boidCount=null,this.simulationSize=null,this.cellSize=50,this.gridDim={x:1,y:1,z:1},this.numCells=1,this.boidBuffer=null,this.cellHeadBuffer=null,this.boidNextBuffer=null,this.matrixBuffer=null,this.matrixStagingBuffer=null,this.uniformBuffer=null,this.clearCellsPipeline=null,this.hashInsertPipeline=null,this.updateBoidsPipeline=null,this.computeMatricesPipeline=null,this.bindGroupLayout=null,this.bindGroup=null,this.isMapping=!1}async init(e,t){this.boidCount=e,this.boidDensity=t;const n=await navigator.gpu?.requestAdapter();if(!n)return console.error("WebGPU not supported"),!1;this.gpuDevice=await n.requestDevice();const r=await fetch("compute-shader.wgsl").then(c=>c.text()),s=this.gpuDevice.createShaderModule({code:r});this.resetParamsToDefaults(e,t),this.initBoidBuffers(),this.initSpatialHashBuffers(),this.initMatrixBuffers(),this.uniformBuffer=this.gpuDevice.createBuffer({size:this.paramsArray.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.syncParams(),this.bindGroupLayout=this.gpuDevice.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});const a=this.gpuDevice.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]}),o=c=>this.gpuDevice.createComputePipeline({layout:a,compute:{module:s,entryPoint:c}});return this.clearCellsPipeline=o("clear_cells"),this.hashInsertPipeline=o("hash_insert"),this.updateBoidsPipeline=o("update_boids"),this.computeMatricesPipeline=o("compute_matrices"),this.createBindGroups(),!0}resetParamsToDefaults(e,t){this.boidCount=e,this.simulationSize=Fl(e,t),this.paramsArray[Be.SEPARATION_DIST]=25,this.paramsArray[Be.ALIGN_DIST]=50,this.paramsArray[Be.COHESION_DIST]=50,this.updateGrid(),this.paramsArray[Be.MAX_SPEED]=5,this.paramsArray[Be.MAX_FORCE]=.1,this.paramsArray[Be.SEPARATION_WEIGHT]=1.5,this.paramsArray[Be.ALIGNMENT_WEIGHT]=1,this.paramsArray[Be.COHESION_WEIGHT]=.5,this.paramsArray[Be.MARGIN]=100,this.paramsArray[Be.TURN_FACTOR]=.2,this.paramsArray[Be.VISION_ANGLE]=Math.PI*1.5,this.updateParamsArrayFromGrid()}updateGrid(){const e=hg(this.paramsArray,this.simulationSize);this.cellSize=e.cellSize,this.gridDim=e.gridDim,this.numCells=e.numCells}updateParamsArrayFromGrid(){this.paramsArray[Be.CELL_SIZE]=this.cellSize,this.paramsArray[Be.PADDING]=0,this.paramsArray[Be.WORLD_MAX]=this.simulationSize.x,this.paramsArray[Be.WORLD_MAX+1]=this.simulationSize.y,this.paramsArray[Be.WORLD_MAX+2]=this.simulationSize.z,this.paramsArray[Be.WORLD_MAX+3]=0,this.paramsArray[Be.GRID_DIM]=this.gridDim.x,this.paramsArray[Be.GRID_DIM+1]=this.gridDim.y,this.paramsArray[Be.GRID_DIM+2]=this.gridDim.z,this.paramsArray[Be.GRID_DIM+3]=this.numCells}syncParams(){!this.gpuDevice||!this.uniformBuffer||(this.updateGrid(),this.updateParamsArrayFromGrid(),this.gpuDevice.queue.writeBuffer(this.uniformBuffer,0,this.paramsArray.buffer,this.paramsArray.byteOffset,this.paramsArray.byteLength))}initBoidBuffers(){this.boidBuffer&&this.boidBuffer.destroy();const e=new Float32Array(this.boidCount*8),t=cg(this.simulationSize);for(let n=0;n<this.boidCount;n++)e[n*8]=t.min.x+Math.random()*(t.max.x-t.min.x),e[n*8+1]=t.min.y+Math.random()*(t.max.y-t.min.y),e[n*8+2]=t.min.z+Math.random()*(t.max.z-t.min.z),e[n*8+3]=1,e[n*8+4]=(Math.random()-.5)*4,e[n*8+5]=(Math.random()-.5)*4,e[n*8+6]=(Math.random()-.5)*4,e[n*8+7]=0;this.boidBuffer=this.gpuDevice.createBuffer({size:e.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.boidBuffer.getMappedRange()).set(e),this.boidBuffer.unmap()}initSpatialHashBuffers(){this.cellHeadBuffer&&this.cellHeadBuffer.destroy(),this.boidNextBuffer&&this.boidNextBuffer.destroy(),this.updateGrid(),this.cellHeadBuffer=this.gpuDevice.createBuffer({size:Math.max(4,this.numCells*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.boidNextBuffer=this.gpuDevice.createBuffer({size:Math.max(4,this.boidCount*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})}initMatrixBuffers(){this.matrixBuffer&&this.matrixBuffer.destroy(),this.matrixStagingBuffer&&this.matrixStagingBuffer.destroy();const e=this.boidCount*16*4;this.matrixBuffer=this.gpuDevice.createBuffer({size:Math.max(4,e),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.matrixStagingBuffer=this.gpuDevice.createBuffer({size:Math.max(4,e),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ})}createBindGroups(){this.bindGroup=this.gpuDevice.createBindGroup({layout:this.bindGroupLayout,entries:[{binding:0,resource:{buffer:this.boidBuffer}},{binding:1,resource:{buffer:this.uniformBuffer}},{binding:2,resource:{buffer:this.cellHeadBuffer}},{binding:3,resource:{buffer:this.boidNextBuffer}},{binding:4,resource:{buffer:this.matrixBuffer}}]})}recreateBoids(e,t){this.boidCount=e,this.simulationSize=Fl(e,t),this.resetParamsToDefaults(e,t),this.initBoidBuffers(),this.initSpatialHashBuffers(),this.initMatrixBuffers(),this.syncParams(),this.createBindGroups()}async step(e){if(!this.gpuDevice||this.isMapping)return null;const t=performance.now();this.syncParams();const n=this.gpuDevice.createCommandEncoder(),r=Math.ceil(this.boidCount/Ul),s=Math.ceil(this.numCells/Ul),a=n.beginComputePass();a.setPipeline(this.clearCellsPipeline),a.setBindGroup(0,this.bindGroup),a.dispatchWorkgroups(s),a.end();const o=n.beginComputePass();o.setPipeline(this.hashInsertPipeline),o.setBindGroup(0,this.bindGroup),o.dispatchWorkgroups(r),o.end();const c=n.beginComputePass();c.setPipeline(this.updateBoidsPipeline),c.setBindGroup(0,this.bindGroup),c.dispatchWorkgroups(r),c.end();const l=n.beginComputePass();l.setPipeline(this.computeMatricesPipeline),l.setBindGroup(0,this.bindGroup),l.dispatchWorkgroups(r),l.end(),n.copyBufferToBuffer(this.matrixBuffer,0,this.matrixStagingBuffer,0,this.matrixBuffer.size),this.gpuDevice.queue.submit([n.finish()]);const u=performance.now()-t;this.isMapping=!0;try{await this.matrixStagingBuffer.mapAsync(GPUMapMode.READ);const d=performance.now(),h=new Float32Array(this.matrixStagingBuffer.getMappedRange());e.updateInstances(h),this.matrixStagingBuffer.unmap(),this.isMapping=!1;const f=performance.now()-d;return{simDelta:u,renderDelta:f}}catch{return this.isMapping=!1,null}}}function dg(i,e=!1){const t=i[0].index!==null,n=new Set(Object.keys(i[0].attributes)),r=new Set(Object.keys(i[0].morphAttributes)),s={},a={},o=i[0].morphTargetsRelative,c=new Qt;let l=0;for(let u=0;u<i.length;++u){const d=i[u];let h=0;if(t!==(d.index!==null))return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+u+". All geometries must have compatible attributes; make sure index attribute exists among all geometries, or in none of them."),null;for(const f in d.attributes){if(!n.has(f))return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+u+'. All geometries must have compatible attributes; make sure "'+f+'" attribute exists among all geometries, or in none of them.'),null;s[f]===void 0&&(s[f]=[]),s[f].push(d.attributes[f]),h++}if(h!==n.size)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+u+". Make sure all geometries have the same number of attributes."),null;if(o!==d.morphTargetsRelative)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+u+". .morphTargetsRelative must be consistent throughout all geometries."),null;for(const f in d.morphAttributes){if(!r.has(f))return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+u+".  .morphAttributes must be consistent throughout all geometries."),null;a[f]===void 0&&(a[f]=[]),a[f].push(d.morphAttributes[f])}if(e){let f;if(t)f=d.index.count;else if(d.attributes.position!==void 0)f=d.attributes.position.count;else return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed with geometry at index "+u+". The geometry must have either an index or a position attribute"),null;c.addGroup(l,f,u),l+=f}}if(t){let u=0;const d=[];for(let h=0;h<i.length;++h){const f=i[h].index;for(let _=0;_<f.count;++_)d.push(f.getX(_)+u);u+=i[h].attributes.position.count}c.setIndex(d)}for(const u in s){const d=Nl(s[u]);if(!d)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed while trying to merge the "+u+" attribute."),null;c.setAttribute(u,d)}for(const u in a){const d=a[u][0].length;if(d===0)break;c.morphAttributes=c.morphAttributes||{},c.morphAttributes[u]=[];for(let h=0;h<d;++h){const f=[];for(let y=0;y<a[u].length;++y)f.push(a[u][y][h]);const _=Nl(f);if(!_)return console.error("THREE.BufferGeometryUtils: .mergeGeometries() failed while trying to merge the "+u+" morphAttribute."),null;c.morphAttributes[u].push(_)}}return c}function Nl(i){let e,t,n,r=-1,s=0;for(let l=0;l<i.length;++l){const u=i[l];if(e===void 0&&(e=u.array.constructor),e!==u.array.constructor)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.array must be of consistent array types across matching attributes."),null;if(t===void 0&&(t=u.itemSize),t!==u.itemSize)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.itemSize must be consistent across matching attributes."),null;if(n===void 0&&(n=u.normalized),n!==u.normalized)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.normalized must be consistent across matching attributes."),null;if(r===-1&&(r=u.gpuType),r!==u.gpuType)return console.error("THREE.BufferGeometryUtils: .mergeAttributes() failed. BufferAttribute.gpuType must be consistent across matching attributes."),null;s+=u.count*t}const a=new e(s),o=new on(a,t,n);let c=0;for(let l=0;l<i.length;++l){const u=i[l];if(u.isInterleavedBufferAttribute){const d=c/t;for(let h=0,f=u.count;h<f;h++)for(let _=0;_<t;_++){const y=u.getComponent(h,_);o.setComponent(h+d,_,y)}}else a.set(u.array,c);c+=u.count*t}return r!==void 0&&(o.gpuType=r),o}const Ol={type:"change"},yo={type:"start"},Sc={type:"end"},ns=new _s,Bl=new Qn,fg=Math.cos(70*fh.DEG2RAD),At=new q,Wt=2*Math.PI,gt={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},na=1e-6;class pg extends uu{constructor(e,t=null){super(e,t),this.state=gt.NONE,this.target=new q,this.cursor=new q,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:zi.ROTATE,MIDDLE:zi.DOLLY,RIGHT:zi.PAN},this.touches={ONE:Bi.ROTATE,TWO:Bi.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._cursorStyle="auto",this._domElementKeyEvents=null,this._lastPosition=new q,this._lastQuaternion=new si,this._lastTargetPosition=new q,this._quat=new si().setFromUnitVectors(e.up,new q(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new cl,this._sphericalDelta=new cl,this._scale=1,this._panOffset=new q,this._rotateStart=new $e,this._rotateEnd=new $e,this._rotateDelta=new $e,this._panStart=new $e,this._panEnd=new $e,this._panDelta=new $e,this._dollyStart=new $e,this._dollyEnd=new $e,this._dollyDelta=new $e,this._dollyDirection=new q,this._mouse=new $e,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=gg.bind(this),this._onPointerDown=mg.bind(this),this._onPointerUp=_g.bind(this),this._onContextMenu=bg.bind(this),this._onMouseWheel=Sg.bind(this),this._onKeyDown=Mg.bind(this),this._onTouchStart=yg.bind(this),this._onTouchMove=Eg.bind(this),this._onMouseDown=xg.bind(this),this._onMouseMove=vg.bind(this),this._interceptControlDown=Tg.bind(this),this._interceptControlUp=Ag.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}set cursorStyle(e){this._cursorStyle=e,e==="grab"?this.domElement.style.cursor="grab":this.domElement.style.cursor="auto"}get cursorStyle(){return this._cursorStyle}connect(e){super.connect(e),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.getRootNode().addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.getRootNode().removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.style.touchAction="auto"}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(e){e.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=e}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(Ol),this.update(),this.state=gt.NONE}pan(e,t){this._pan(e,t),this.update()}dollyIn(e){this._dollyIn(e),this.update()}dollyOut(e){this._dollyOut(e),this.update()}rotateLeft(e){this._rotateLeft(e),this.update()}rotateUp(e){this._rotateUp(e),this.update()}update(e=null){const t=this.object.position;At.copy(t).sub(this.target),At.applyQuaternion(this._quat),this._spherical.setFromVector3(At),this.autoRotate&&this.state===gt.NONE&&this._rotateLeft(this._getAutoRotationAngle(e)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let n=this.minAzimuthAngle,r=this.maxAzimuthAngle;isFinite(n)&&isFinite(r)&&(n<-Math.PI?n+=Wt:n>Math.PI&&(n-=Wt),r<-Math.PI?r+=Wt:r>Math.PI&&(r-=Wt),n<=r?this._spherical.theta=Math.max(n,Math.min(r,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(n+r)/2?Math.max(n,this._spherical.theta):Math.min(r,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let s=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{const a=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),s=a!=this._spherical.radius}if(At.setFromSpherical(this._spherical),At.applyQuaternion(this._quatInverse),t.copy(this.target).add(At),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let a=null;if(this.object.isPerspectiveCamera){const o=At.length();a=this._clampDistance(o*this._scale);const c=o-a;this.object.position.addScaledVector(this._dollyDirection,c),this.object.updateMatrixWorld(),s=!!c}else if(this.object.isOrthographicCamera){const o=new q(this._mouse.x,this._mouse.y,0);o.unproject(this.object);const c=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),s=c!==this.object.zoom;const l=new q(this._mouse.x,this._mouse.y,0);l.unproject(this.object),this.object.position.sub(l).add(o),this.object.updateMatrixWorld(),a=At.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;a!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(a).add(this.object.position):(ns.origin.copy(this.object.position),ns.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(ns.direction))<fg?this.object.lookAt(this.target):(Bl.setFromNormalAndCoplanarPoint(this.object.up,this.target),ns.intersectPlane(Bl,this.target))))}else if(this.object.isOrthographicCamera){const a=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),a!==this.object.zoom&&(this.object.updateProjectionMatrix(),s=!0)}return this._scale=1,this._performCursorZoom=!1,s||this._lastPosition.distanceToSquared(this.object.position)>na||8*(1-this._lastQuaternion.dot(this.object.quaternion))>na||this._lastTargetPosition.distanceToSquared(this.target)>na?(this.dispatchEvent(Ol),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(e){return e!==null?Wt/60*this.autoRotateSpeed*e:Wt/60/60*this.autoRotateSpeed}_getZoomScale(e){const t=Math.abs(e*.01);return Math.pow(.95,this.zoomSpeed*t)}_rotateLeft(e){this._sphericalDelta.theta-=e}_rotateUp(e){this._sphericalDelta.phi-=e}_panLeft(e,t){At.setFromMatrixColumn(t,0),At.multiplyScalar(-e),this._panOffset.add(At)}_panUp(e,t){this.screenSpacePanning===!0?At.setFromMatrixColumn(t,1):(At.setFromMatrixColumn(t,0),At.crossVectors(this.object.up,At)),At.multiplyScalar(e),this._panOffset.add(At)}_pan(e,t){const n=this.domElement;if(this.object.isPerspectiveCamera){const r=this.object.position;At.copy(r).sub(this.target);let s=At.length();s*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*e*s/n.clientHeight,this.object.matrix),this._panUp(2*t*s/n.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(e*(this.object.right-this.object.left)/this.object.zoom/n.clientWidth,this.object.matrix),this._panUp(t*(this.object.top-this.object.bottom)/this.object.zoom/n.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(e){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=e:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(e,t){if(!this.zoomToCursor)return;this._performCursorZoom=!0;const n=this.domElement.getBoundingClientRect(),r=e-n.left,s=t-n.top,a=n.width,o=n.height;this._mouse.x=r/a*2-1,this._mouse.y=-(s/o)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(e){return Math.max(this.minDistance,Math.min(this.maxDistance,e))}_handleMouseDownRotate(e){this._rotateStart.set(e.clientX,e.clientY)}_handleMouseDownDolly(e){this._updateZoomParameters(e.clientX,e.clientX),this._dollyStart.set(e.clientX,e.clientY)}_handleMouseDownPan(e){this._panStart.set(e.clientX,e.clientY)}_handleMouseMoveRotate(e){this._rotateEnd.set(e.clientX,e.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(Wt*this._rotateDelta.x/t.clientHeight),this._rotateUp(Wt*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(e){this._dollyEnd.set(e.clientX,e.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.update()}_handleMouseMovePan(e){this._panEnd.set(e.clientX,e.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(e){this._updateZoomParameters(e.clientX,e.clientY),e.deltaY<0?this._dollyIn(this._getZoomScale(e.deltaY)):e.deltaY>0&&this._dollyOut(this._getZoomScale(e.deltaY)),this.update()}_handleKeyDown(e){let t=!1;switch(e.code){case this.keys.UP:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(Wt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),t=!0;break;case this.keys.BOTTOM:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateUp(-Wt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),t=!0;break;case this.keys.LEFT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(Wt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),t=!0;break;case this.keys.RIGHT:e.ctrlKey||e.metaKey||e.shiftKey?this.enableRotate&&this._rotateLeft(-Wt*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),t=!0;break}t&&(e.preventDefault(),this.update())}_handleTouchStartRotate(e){if(this._pointers.length===1)this._rotateStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),r=.5*(e.pageY+t.y);this._rotateStart.set(n,r)}}_handleTouchStartPan(e){if(this._pointers.length===1)this._panStart.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),r=.5*(e.pageY+t.y);this._panStart.set(n,r)}}_handleTouchStartDolly(e){const t=this._getSecondPointerPosition(e),n=e.pageX-t.x,r=e.pageY-t.y,s=Math.sqrt(n*n+r*r);this._dollyStart.set(0,s)}_handleTouchStartDollyPan(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enablePan&&this._handleTouchStartPan(e)}_handleTouchStartDollyRotate(e){this.enableZoom&&this._handleTouchStartDolly(e),this.enableRotate&&this._handleTouchStartRotate(e)}_handleTouchMoveRotate(e){if(this._pointers.length==1)this._rotateEnd.set(e.pageX,e.pageY);else{const n=this._getSecondPointerPosition(e),r=.5*(e.pageX+n.x),s=.5*(e.pageY+n.y);this._rotateEnd.set(r,s)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);const t=this.domElement;this._rotateLeft(Wt*this._rotateDelta.x/t.clientHeight),this._rotateUp(Wt*this._rotateDelta.y/t.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(e){if(this._pointers.length===1)this._panEnd.set(e.pageX,e.pageY);else{const t=this._getSecondPointerPosition(e),n=.5*(e.pageX+t.x),r=.5*(e.pageY+t.y);this._panEnd.set(n,r)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(e){const t=this._getSecondPointerPosition(e),n=e.pageX-t.x,r=e.pageY-t.y,s=Math.sqrt(n*n+r*r);this._dollyEnd.set(0,s),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);const a=(e.pageX+t.x)*.5,o=(e.pageY+t.y)*.5;this._updateZoomParameters(a,o)}_handleTouchMoveDollyPan(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enablePan&&this._handleTouchMovePan(e)}_handleTouchMoveDollyRotate(e){this.enableZoom&&this._handleTouchMoveDolly(e),this.enableRotate&&this._handleTouchMoveRotate(e)}_addPointer(e){this._pointers.push(e.pointerId)}_removePointer(e){delete this._pointerPositions[e.pointerId];for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId){this._pointers.splice(t,1);return}}_isTrackingPointer(e){for(let t=0;t<this._pointers.length;t++)if(this._pointers[t]==e.pointerId)return!0;return!1}_trackPointer(e){let t=this._pointerPositions[e.pointerId];t===void 0&&(t=new $e,this._pointerPositions[e.pointerId]=t),t.set(e.pageX,e.pageY)}_getSecondPointerPosition(e){const t=e.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[t]}_customWheelEvent(e){const t=e.deltaMode,n={clientX:e.clientX,clientY:e.clientY,deltaY:e.deltaY};switch(t){case 1:n.deltaY*=16;break;case 2:n.deltaY*=100;break}return e.ctrlKey&&!this._controlActive&&(n.deltaY*=10),n}}function mg(i){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(i.pointerId),this.domElement.ownerDocument.addEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(i)&&(this._addPointer(i),i.pointerType==="touch"?this._onTouchStart(i):this._onMouseDown(i),this._cursorStyle==="grab"&&(this.domElement.style.cursor="grabbing")))}function gg(i){this.enabled!==!1&&(i.pointerType==="touch"?this._onTouchMove(i):this._onMouseMove(i))}function _g(i){switch(this._removePointer(i),this._pointers.length){case 0:this.domElement.releasePointerCapture(i.pointerId),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(Sc),this.state=gt.NONE,this._cursorStyle==="grab"&&(this.domElement.style.cursor="grab");break;case 1:const e=this._pointers[0],t=this._pointerPositions[e];this._onTouchStart({pointerId:e,pageX:t.x,pageY:t.y});break}}function xg(i){let e;switch(i.button){case 0:e=this.mouseButtons.LEFT;break;case 1:e=this.mouseButtons.MIDDLE;break;case 2:e=this.mouseButtons.RIGHT;break;default:e=-1}switch(e){case zi.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(i),this.state=gt.DOLLY;break;case zi.ROTATE:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=gt.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=gt.ROTATE}break;case zi.PAN:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=gt.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=gt.PAN}break;default:this.state=gt.NONE}this.state!==gt.NONE&&this.dispatchEvent(yo)}function vg(i){switch(this.state){case gt.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(i);break;case gt.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(i);break;case gt.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(i);break}}function Sg(i){this.enabled===!1||this.enableZoom===!1||this.state!==gt.NONE||(i.preventDefault(),this.dispatchEvent(yo),this._handleMouseWheel(this._customWheelEvent(i)),this.dispatchEvent(Sc))}function Mg(i){this.enabled!==!1&&this._handleKeyDown(i)}function yg(i){switch(this._trackPointer(i),this._pointers.length){case 1:switch(this.touches.ONE){case Bi.ROTATE:if(this.enableRotate===!1)return;this._handleTouchStartRotate(i),this.state=gt.TOUCH_ROTATE;break;case Bi.PAN:if(this.enablePan===!1)return;this._handleTouchStartPan(i),this.state=gt.TOUCH_PAN;break;default:this.state=gt.NONE}break;case 2:switch(this.touches.TWO){case Bi.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(i),this.state=gt.TOUCH_DOLLY_PAN;break;case Bi.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(i),this.state=gt.TOUCH_DOLLY_ROTATE;break;default:this.state=gt.NONE}break;default:this.state=gt.NONE}this.state!==gt.NONE&&this.dispatchEvent(yo)}function Eg(i){switch(this._trackPointer(i),this.state){case gt.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(i),this.update();break;case gt.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(i),this.update();break;case gt.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(i),this.update();break;case gt.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(i),this.update();break;default:this.state=gt.NONE}}function bg(i){this.enabled!==!1&&i.preventDefault()}function Tg(i){i.key==="Control"&&(this._controlActive=!0,this.domElement.getRootNode().addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function Ag(i){i.key==="Control"&&(this._controlActive=!1,this.domElement.getRootNode().removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}var an=Uint8Array,ki=Uint16Array,wg=Int32Array,Mc=new an([0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0,0]),yc=new an([0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,0,0]),Cg=new an([16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]),Ec=function(i,e){for(var t=new ki(31),n=0;n<31;++n)t[n]=e+=1<<i[n-1];for(var r=new wg(t[30]),n=1;n<30;++n)for(var s=t[n];s<t[n+1];++s)r[s]=s-t[n]<<5|n;return{b:t,r}},bc=Ec(Mc,2),Tc=bc.b,Rg=bc.r;Tc[28]=258,Rg[258]=28;var Pg=Ec(yc,0),Dg=Pg.b,eo=new ki(32768);for(var vt=0;vt<32768;++vt){var Kn=(vt&43690)>>1|(vt&21845)<<1;Kn=(Kn&52428)>>2|(Kn&13107)<<2,Kn=(Kn&61680)>>4|(Kn&3855)<<4,eo[vt]=((Kn&65280)>>8|(Kn&255)<<8)>>1}var pr=(function(i,e,t){for(var n=i.length,r=0,s=new ki(e);r<n;++r)i[r]&&++s[i[r]-1];var a=new ki(e);for(r=1;r<e;++r)a[r]=a[r-1]+s[r-1]<<1;var o;if(t){o=new ki(1<<e);var c=15-e;for(r=0;r<n;++r)if(i[r])for(var l=r<<4|i[r],u=e-i[r],d=a[i[r]-1]++<<u,h=d|(1<<u)-1;d<=h;++d)o[eo[d]>>c]=l}else for(o=new ki(n),r=0;r<n;++r)i[r]&&(o[r]=eo[a[i[r]-1]++]>>15-i[r]);return o}),Sr=new an(288);for(var vt=0;vt<144;++vt)Sr[vt]=8;for(var vt=144;vt<256;++vt)Sr[vt]=9;for(var vt=256;vt<280;++vt)Sr[vt]=7;for(var vt=280;vt<288;++vt)Sr[vt]=8;var Ac=new an(32);for(var vt=0;vt<32;++vt)Ac[vt]=5;var Ig=pr(Sr,9,1),Lg=pr(Ac,5,1),ia=function(i){for(var e=i[0],t=1;t<i.length;++t)i[t]>e&&(e=i[t]);return e},fn=function(i,e,t){var n=e/8|0;return(i[n]|i[n+1]<<8)>>(e&7)&t},ra=function(i,e){var t=e/8|0;return(i[t]|i[t+1]<<8|i[t+2]<<16)>>(e&7)},Ug=function(i){return(i+7)/8|0},Fg=function(i,e,t){return(t==null||t>i.length)&&(t=i.length),new an(i.subarray(e,t))},Ng=["unexpected EOF","invalid block type","invalid length/literal","invalid distance","stream finished","no stream handler",,"no callback","invalid UTF-8 data","extra field too long","date not in range 1980-2099","filename too long","stream finishing","invalid zip data"],pn=function(i,e,t){var n=new Error(e||Ng[i]);if(n.code=i,Error.captureStackTrace&&Error.captureStackTrace(n,pn),!t)throw n;return n},Og=function(i,e,t,n){var r=i.length,s=0;if(!r||e.f&&!e.l)return t||new an(0);var a=!t,o=a||e.i!=2,c=e.i;a&&(t=new an(r*3));var l=function(Fe){var xt=t.length;if(Fe>xt){var et=new an(Math.max(xt*2,Fe));et.set(t),t=et}},u=e.f||0,d=e.p||0,h=e.b||0,f=e.l,_=e.d,y=e.m,g=e.n,m=r*8;do{if(!f){u=fn(i,d,1);var b=fn(i,d+1,3);if(d+=3,b)if(b==1)f=Ig,_=Lg,y=9,g=5;else if(b==2){var L=fn(i,d,31)+257,N=fn(i,d+10,15)+4,S=L+fn(i,d+5,31)+1;d+=14;for(var T=new an(S),G=new an(19),D=0;D<N;++D)G[Cg[D]]=fn(i,d+D*3,7);d+=N*3;for(var O=ia(G),V=(1<<O)-1,K=pr(G,O,1),D=0;D<S;){var Y=K[fn(i,d,V)];d+=Y&15;var w=Y>>4;if(w<16)T[D++]=w;else{var Z=0,X=0;for(w==16?(X=3+fn(i,d,3),d+=2,Z=T[D-1]):w==17?(X=3+fn(i,d,7),d+=3):w==18&&(X=11+fn(i,d,127),d+=7);X--;)T[D++]=Z}}var fe=T.subarray(0,L),oe=T.subarray(L);y=ia(fe),g=ia(oe),f=pr(fe,y,1),_=pr(oe,g,1)}else pn(1);else{var w=Ug(d)+4,A=i[w-4]|i[w-3]<<8,U=w+A;if(U>r){c&&pn(0);break}o&&l(h+A),t.set(i.subarray(w,U),h),e.b=h+=A,e.p=d=U*8,e.f=u;continue}if(d>m){c&&pn(0);break}}o&&l(h+131072);for(var ye=(1<<y)-1,Ae=(1<<g)-1,ve=d;;ve=d){var Z=f[ra(i,d)&ye],Ge=Z>>4;if(d+=Z&15,d>m){c&&pn(0);break}if(Z||pn(2),Ge<256)t[h++]=Ge;else if(Ge==256){ve=d,f=null;break}else{var st=Ge-254;if(Ge>264){var D=Ge-257,_e=Mc[D];st=fn(i,d,(1<<_e)-1)+Tc[D],d+=_e}var $=_[ra(i,d)&Ae],ue=$>>4;$||pn(3),d+=$&15;var oe=Dg[ue];if(ue>3){var _e=yc[ue];oe+=ra(i,d)&(1<<_e)-1,d+=_e}if(d>m){c&&pn(0);break}o&&l(h+131072);var de=h+st;if(h<oe){var ze=s-oe,Le=Math.min(oe,de);for(ze+h<0&&pn(3);h<Le;++h)t[h]=n[ze+h]}for(;h<de;++h)t[h]=t[h-oe]}}e.l=f,e.p=ve,e.b=h,e.f=u,f&&(u=1,e.m=y,e.d=_,e.n=g)}while(!u);return h!=t.length&&a?Fg(t,0,h):t.subarray(0,h)},Bg=new an(0),kg=function(i,e){return((i[0]&15)!=8||i[0]>>4>7||(i[0]<<8|i[1])%31)&&pn(6,"invalid zlib data"),(i[1]>>5&1)==1&&pn(6,"invalid zlib data: "+(i[1]&32?"need":"unexpected")+" dictionary"),(i[1]>>3&4)+2};function is(i,e){return Og(i.subarray(kg(i),-4),{i:2},e,e)}var zg=typeof TextDecoder<"u"&&new TextDecoder,Gg=0;try{zg.decode(Bg,{stream:!0}),Gg=1}catch{}class Hg extends iu{constructor(e){super(e),this.type=Jt,this.outputFormat=Ft}parse(e){const T=Math.pow(2.7182818,2.2);function G(p,E){let I=0;for(let ie=0;ie<65536;++ie)(ie==0||p[ie>>3]&1<<(ie&7))&&(E[I++]=ie);const H=I-1;for(;I<65536;)E[I++]=0;return H}function D(p){for(let E=0;E<16384;E++)p[E]={},p[E].len=0,p[E].lit=0,p[E].p=null}const O={l:0,c:0,lc:0};function V(p,E,I,H,ie){for(;I<p;)E=E<<8|Re(H,ie),I+=8;I-=p,O.l=E>>I&(1<<p)-1,O.c=E,O.lc=I}const K=new Array(59);function Y(p){for(let I=0;I<=58;++I)K[I]=0;for(let I=0;I<65537;++I)K[p[I]]+=1;let E=0;for(let I=58;I>0;--I){const H=E+K[I]>>1;K[I]=E,E=H}for(let I=0;I<65537;++I){const H=p[I];H>0&&(p[I]=H|K[H]++<<6)}}function Z(p,E,I,H,ie,x){const M=E;let F=0,P=0;for(;H<=ie;H++){if(M.value-E.value>I)return!1;V(6,F,P,p,M);const R=O.l;if(F=O.c,P=O.lc,x[H]=R,R==63){if(M.value-E.value>I)throw new Error("Something wrong with hufUnpackEncTable");V(8,F,P,p,M);let W=O.l+6;if(F=O.c,P=O.lc,H+W>ie+1)throw new Error("Something wrong with hufUnpackEncTable");for(;W--;)x[H++]=0;H--}else if(R>=59){let W=R-59+2;if(H+W>ie+1)throw new Error("Something wrong with hufUnpackEncTable");for(;W--;)x[H++]=0;H--}}Y(x)}function X(p){return p&63}function fe(p){return p>>6}function oe(p,E,I,H){for(;E<=I;E++){const ie=fe(p[E]),x=X(p[E]);if(ie>>x)throw new Error("Invalid table entry");if(x>14){const M=H[ie>>x-14];if(M.len)throw new Error("Invalid table entry");if(M.lit++,M.p){const F=M.p;M.p=new Array(M.lit);for(let P=0;P<M.lit-1;++P)M.p[P]=F[P]}else M.p=new Array(1);M.p[M.lit-1]=E}else if(x){let M=0;for(let F=1<<14-x;F>0;F--){const P=H[(ie<<14-x)+M];if(P.len||P.p)throw new Error("Invalid table entry");P.len=x,P.lit=E,M++}}}return!0}const ye={c:0,lc:0};function Ae(p,E,I,H){p=p<<8|Re(I,H),E+=8,ye.c=p,ye.lc=E}const ve={c:0,lc:0};function Ge(p,E,I,H,ie,x,M,F,P){if(p==E){H<8&&(Ae(I,H,ie,x),I=ye.c,H=ye.lc),H-=8;let R=I>>H;if(R=new Uint8Array([R])[0],F.value+R>P)return!1;const W=M[F.value-1];for(;R-- >0;)M[F.value++]=W}else if(F.value<P)M[F.value++]=p;else return!1;ve.c=I,ve.lc=H}function st(p){return p&65535}function _e(p){const E=st(p);return E>32767?E-65536:E}const $={a:0,b:0};function ue(p,E){const I=_e(p),ie=_e(E),x=I+(ie&1)+(ie>>1),M=x,F=x-ie;$.a=M,$.b=F}function de(p,E){const I=st(p),H=st(E),ie=I-(H>>1)&65535,x=H+ie-32768&65535;$.a=x,$.b=ie}function ze(p,E,I,H,ie,x,M){const F=M<16384,P=I>ie?ie:I;let R=1,W,Q;for(;R<=P;)R<<=1;for(R>>=1,W=R,R>>=1;R>=1;){Q=0;const j=Q+x*(ie-W),ee=x*R,ae=x*W,se=H*R,he=H*W;let ce,Ie,We,je;for(;Q<=j;Q+=ae){let Oe=Q;const Ue=Q+H*(I-W);for(;Oe<=Ue;Oe+=he){const Me=Oe+se,ht=Oe+ee,Ne=ht+se;F?(ue(p[Oe+E],p[ht+E]),ce=$.a,We=$.b,ue(p[Me+E],p[Ne+E]),Ie=$.a,je=$.b,ue(ce,Ie),p[Oe+E]=$.a,p[Me+E]=$.b,ue(We,je),p[ht+E]=$.a,p[Ne+E]=$.b):(de(p[Oe+E],p[ht+E]),ce=$.a,We=$.b,de(p[Me+E],p[Ne+E]),Ie=$.a,je=$.b,de(ce,Ie),p[Oe+E]=$.a,p[Me+E]=$.b,de(We,je),p[ht+E]=$.a,p[Ne+E]=$.b)}if(I&R){const Me=Oe+ee;F?ue(p[Oe+E],p[Me+E]):de(p[Oe+E],p[Me+E]),ce=$.a,p[Me+E]=$.b,p[Oe+E]=ce}}if(ie&R){let Oe=Q;const Ue=Q+H*(I-W);for(;Oe<=Ue;Oe+=he){const Me=Oe+se;F?ue(p[Oe+E],p[Me+E]):de(p[Oe+E],p[Me+E]),ce=$.a,p[Me+E]=$.b,p[Oe+E]=ce}}W=R,R>>=1}return Q}function Le(p,E,I,H,ie,x,M,F,P){let R=0,W=0;const Q=M,j=Math.trunc(H.value+(ie+7)/8);for(;H.value<j;)for(Ae(R,W,I,H),R=ye.c,W=ye.lc;W>=14;){const ae=R>>W-14&16383,se=E[ae];if(se.len)W-=se.len,Ge(se.lit,x,R,W,I,H,F,P,Q),R=ve.c,W=ve.lc;else{if(!se.p)throw new Error("hufDecode issues");let he;for(he=0;he<se.lit;he++){const ce=X(p[se.p[he]]);for(;W<ce&&H.value<j;)Ae(R,W,I,H),R=ye.c,W=ye.lc;if(W>=ce&&fe(p[se.p[he]])==(R>>W-ce&(1<<ce)-1)){W-=ce,Ge(se.p[he],x,R,W,I,H,F,P,Q),R=ve.c,W=ve.lc;break}}if(he==se.lit)throw new Error("hufDecode issues")}}const ee=8-ie&7;for(R>>=ee,W-=ee;W>0;){const ae=E[R<<14-W&16383];if(ae.len)W-=ae.len,Ge(ae.lit,x,R,W,I,H,F,P,Q),R=ve.c,W=ve.lc;else throw new Error("hufDecode issues")}return!0}function Fe(p,E,I,H,ie,x){const M={value:0},F=I.value,P=Te(E,I),R=Te(E,I);I.value+=4;const W=Te(E,I);if(I.value+=4,P<0||P>=65537||R<0||R>=65537)throw new Error("Something wrong with HUF_ENCSIZE");const Q=new Array(65537),j=new Array(16384);D(j);const ee=H-(I.value-F);if(Z(p,I,ee,P,R,Q),W>8*(H-(I.value-F)))throw new Error("Something wrong with hufUncompress");oe(Q,P,R,j),Le(Q,j,p,I,W,R,x,ie,M)}function xt(p,E,I){for(let H=0;H<I;++H)E[H]=p[E[H]]}function et(p){for(let E=1;E<p.length;E++){const I=p[E-1]+p[E]-128;p[E]=I}}function ct(p,E){let I=0,H=Math.floor((p.length+1)/2),ie=0;const x=p.length-1;for(;!(ie>x||(E[ie++]=p[I++],ie>x));)E[ie++]=p[H++]}function dt(p){let E=p.byteLength;const I=new Array;let H=0;const ie=new DataView(p);for(;E>0;){const x=ie.getInt8(H++);if(x<0){const M=-x;E-=M+1;for(let F=0;F<M;F++)I.push(ie.getUint8(H++))}else{const M=x;E-=2;const F=ie.getUint8(H++);for(let P=0;P<M+1;P++)I.push(F)}}return I}function Ke(p,E,I,H,ie,x){let M=new DataView(x.buffer);const F=I[p.idx[0]].width,P=I[p.idx[0]].height,R=3,W=Math.floor(F/8),Q=Math.ceil(F/8),j=Math.ceil(P/8),ee=F-(Q-1)*8,ae=P-(j-1)*8,se={value:0},he=new Array(R),ce=new Array(R),Ie=new Array(R),We=new Array(R),je=new Array(R);for(let Ue=0;Ue<R;++Ue)je[Ue]=E[p.idx[Ue]],he[Ue]=Ue<1?0:he[Ue-1]+Q*j,ce[Ue]=new Float32Array(64),Ie[Ue]=new Uint16Array(64),We[Ue]=new Uint16Array(Q*64);for(let Ue=0;Ue<j;++Ue){let Me=8;Ue==j-1&&(Me=ae);let ht=8;for(let Ze=0;Ze<Q;++Ze){Ze==Q-1&&(ht=ee);for(let it=0;it<R;++it)Ie[it].fill(0),Ie[it][0]=ie[he[it]++],B(se,H,Ie[it]),Mt(Ie[it],ce[it]),at(ce[it]);pt(ce);for(let it=0;it<R;++it)De(ce[it],We[it],Ze*64)}let Ne=0;for(let Ze=0;Ze<R;++Ze){const it=I[p.idx[Ze]].type;for(let Lt=8*Ue;Lt<8*Ue+Me;++Lt){Ne=je[Ze][Lt];for(let en=0;en<W;++en){const tt=en*64+(Lt&7)*8;M.setUint16(Ne+0*it,We[Ze][tt+0],!0),M.setUint16(Ne+2*it,We[Ze][tt+1],!0),M.setUint16(Ne+4*it,We[Ze][tt+2],!0),M.setUint16(Ne+6*it,We[Ze][tt+3],!0),M.setUint16(Ne+8*it,We[Ze][tt+4],!0),M.setUint16(Ne+10*it,We[Ze][tt+5],!0),M.setUint16(Ne+12*it,We[Ze][tt+6],!0),M.setUint16(Ne+14*it,We[Ze][tt+7],!0),Ne+=16*it}}if(W!=Q)for(let Lt=8*Ue;Lt<8*Ue+Me;++Lt){const en=je[Ze][Lt]+8*W*2*it,tt=W*64+(Lt&7)*8;for(let yt=0;yt<ht;++yt)M.setUint16(en+yt*2*it,We[Ze][tt+yt],!0)}}}const Oe=new Uint16Array(F);M=new DataView(x.buffer);for(let Ue=0;Ue<R;++Ue){I[p.idx[Ue]].decoded=!0;const Me=I[p.idx[Ue]].type;if(I[Ue].type==2)for(let ht=0;ht<P;++ht){const Ne=je[Ue][ht];for(let Ze=0;Ze<F;++Ze)Oe[Ze]=M.getUint16(Ne+Ze*2*Me,!0);for(let Ze=0;Ze<F;++Ze)M.setFloat32(Ne+Ze*2*Me,J(Oe[Ze]),!0)}}}function St(p,E,I,H,ie,x){const M=new DataView(x.buffer),F=I[p],P=F.width,R=F.height,W=Math.ceil(P/8),Q=Math.ceil(R/8),j=Math.floor(P/8),ee=P-(W-1)*8,ae=R-(Q-1)*8,se={value:0};let he=0;const ce=new Float32Array(64),Ie=new Uint16Array(64),We=new Uint16Array(W*64);for(let je=0;je<Q;++je){let Oe=8;je==Q-1&&(Oe=ae);for(let Ue=0;Ue<W;++Ue)Ie.fill(0),Ie[0]=ie[he++],B(se,H,Ie),Mt(Ie,ce),at(ce),De(ce,We,Ue*64);for(let Ue=8*je;Ue<8*je+Oe;++Ue){let Me=E[p][Ue];for(let ht=0;ht<j;++ht){const Ne=ht*64+(Ue&7)*8;for(let Ze=0;Ze<8;++Ze)M.setUint16(Me+Ze*2*F.type,We[Ne+Ze],!0);Me+=16*F.type}if(W!=j){const ht=j*64+(Ue&7)*8;for(let Ne=0;Ne<ee;++Ne)M.setUint16(Me+Ne*2*F.type,We[ht+Ne],!0)}}}F.decoded=!0}function B(p,E,I){let H,ie=1;for(;ie<64;)H=E[p.value],H==65280?ie=64:H>>8==255?ie+=H&255:(I[ie]=H,ie++),p.value++}function Mt(p,E){E[0]=J(p[0]),E[1]=J(p[1]),E[2]=J(p[5]),E[3]=J(p[6]),E[4]=J(p[14]),E[5]=J(p[15]),E[6]=J(p[27]),E[7]=J(p[28]),E[8]=J(p[2]),E[9]=J(p[4]),E[10]=J(p[7]),E[11]=J(p[13]),E[12]=J(p[16]),E[13]=J(p[26]),E[14]=J(p[29]),E[15]=J(p[42]),E[16]=J(p[3]),E[17]=J(p[8]),E[18]=J(p[12]),E[19]=J(p[17]),E[20]=J(p[25]),E[21]=J(p[30]),E[22]=J(p[41]),E[23]=J(p[43]),E[24]=J(p[9]),E[25]=J(p[11]),E[26]=J(p[18]),E[27]=J(p[24]),E[28]=J(p[31]),E[29]=J(p[40]),E[30]=J(p[44]),E[31]=J(p[53]),E[32]=J(p[10]),E[33]=J(p[19]),E[34]=J(p[23]),E[35]=J(p[32]),E[36]=J(p[39]),E[37]=J(p[45]),E[38]=J(p[52]),E[39]=J(p[54]),E[40]=J(p[20]),E[41]=J(p[22]),E[42]=J(p[33]),E[43]=J(p[38]),E[44]=J(p[46]),E[45]=J(p[51]),E[46]=J(p[55]),E[47]=J(p[60]),E[48]=J(p[21]),E[49]=J(p[34]),E[50]=J(p[37]),E[51]=J(p[47]),E[52]=J(p[50]),E[53]=J(p[56]),E[54]=J(p[59]),E[55]=J(p[61]),E[56]=J(p[35]),E[57]=J(p[36]),E[58]=J(p[48]),E[59]=J(p[49]),E[60]=J(p[57]),E[61]=J(p[58]),E[62]=J(p[62]),E[63]=J(p[63])}function at(p){const E=.5*Math.cos(.7853975),I=.5*Math.cos(3.14159/16),H=.5*Math.cos(3.14159/8),ie=.5*Math.cos(3*3.14159/16),x=.5*Math.cos(5*3.14159/16),M=.5*Math.cos(3*3.14159/8),F=.5*Math.cos(7*3.14159/16),P=new Array(4),R=new Array(4),W=new Array(4),Q=new Array(4);for(let j=0;j<8;++j){const ee=j*8;P[0]=H*p[ee+2],P[1]=M*p[ee+2],P[2]=H*p[ee+6],P[3]=M*p[ee+6],R[0]=I*p[ee+1]+ie*p[ee+3]+x*p[ee+5]+F*p[ee+7],R[1]=ie*p[ee+1]-F*p[ee+3]-I*p[ee+5]-x*p[ee+7],R[2]=x*p[ee+1]-I*p[ee+3]+F*p[ee+5]+ie*p[ee+7],R[3]=F*p[ee+1]-x*p[ee+3]+ie*p[ee+5]-I*p[ee+7],W[0]=E*(p[ee+0]+p[ee+4]),W[3]=E*(p[ee+0]-p[ee+4]),W[1]=P[0]+P[3],W[2]=P[1]-P[2],Q[0]=W[0]+W[1],Q[1]=W[3]+W[2],Q[2]=W[3]-W[2],Q[3]=W[0]-W[1],p[ee+0]=Q[0]+R[0],p[ee+1]=Q[1]+R[1],p[ee+2]=Q[2]+R[2],p[ee+3]=Q[3]+R[3],p[ee+4]=Q[3]-R[3],p[ee+5]=Q[2]-R[2],p[ee+6]=Q[1]-R[1],p[ee+7]=Q[0]-R[0]}for(let j=0;j<8;++j)P[0]=H*p[16+j],P[1]=M*p[16+j],P[2]=H*p[48+j],P[3]=M*p[48+j],R[0]=I*p[8+j]+ie*p[24+j]+x*p[40+j]+F*p[56+j],R[1]=ie*p[8+j]-F*p[24+j]-I*p[40+j]-x*p[56+j],R[2]=x*p[8+j]-I*p[24+j]+F*p[40+j]+ie*p[56+j],R[3]=F*p[8+j]-x*p[24+j]+ie*p[40+j]-I*p[56+j],W[0]=E*(p[j]+p[32+j]),W[3]=E*(p[j]-p[32+j]),W[1]=P[0]+P[3],W[2]=P[1]-P[2],Q[0]=W[0]+W[1],Q[1]=W[3]+W[2],Q[2]=W[3]-W[2],Q[3]=W[0]-W[1],p[0+j]=Q[0]+R[0],p[8+j]=Q[1]+R[1],p[16+j]=Q[2]+R[2],p[24+j]=Q[3]+R[3],p[32+j]=Q[3]-R[3],p[40+j]=Q[2]-R[2],p[48+j]=Q[1]-R[1],p[56+j]=Q[0]-R[0]}function pt(p){for(let E=0;E<64;++E){const I=p[0][E],H=p[1][E],ie=p[2][E];p[0][E]=I+1.5747*ie,p[1][E]=I-.1873*H-.4682*ie,p[2][E]=I+1.8556*H}}function De(p,E,I){for(let H=0;H<64;++H)E[I+H]=Wo.toHalfFloat(C(p[H]))}function C(p){return p<=1?Math.sign(p)*Math.pow(Math.abs(p),2.2):Math.sign(p)*Math.pow(T,Math.abs(p)-1)}function v(p){return new DataView(p.array.buffer,p.offset.value,p.size)}function z(p){const E=p.viewer.buffer.slice(p.offset.value,p.offset.value+p.size),I=new Uint8Array(dt(E)),H=new Uint8Array(I.length);return et(I),ct(I,H),new DataView(H.buffer)}function re(p){const E=p.array.slice(p.offset.value,p.offset.value+p.size),I=is(E),H=new Uint8Array(I.length);return et(I),ct(I,H),new DataView(H.buffer)}function le(p){const E=p.viewer,I={value:p.offset.value},H=new Uint16Array(p.columns*p.lines*(p.inputChannels.length*p.type)),ie=new Uint8Array(8192);let x=0;const M=new Array(p.inputChannels.length);for(let ae=0,se=p.inputChannels.length;ae<se;ae++)M[ae]={},M[ae].start=x,M[ae].end=M[ae].start,M[ae].nx=p.columns,M[ae].ny=p.lines,M[ae].size=p.type,x+=M[ae].nx*M[ae].ny*M[ae].size;const F=we(E,I),P=we(E,I);if(P>=8192)throw new Error("Something is wrong with PIZ_COMPRESSION BITMAP_SIZE");if(F<=P)for(let ae=0;ae<P-F+1;ae++)ie[ae+F]=Ee(E,I);const R=new Uint16Array(65536),W=G(ie,R),Q=Te(E,I);Fe(p.array,E,I,Q,H,x);for(let ae=0;ae<p.inputChannels.length;++ae){const se=M[ae];for(let he=0;he<M[ae].size;++he)ze(H,se.start+he,se.nx,se.size,se.ny,se.nx*se.size,W)}xt(R,H,x);let j=0;const ee=new Uint8Array(H.buffer.byteLength);for(let ae=0;ae<p.lines;ae++)for(let se=0;se<p.inputChannels.length;se++){const he=M[se],ce=he.nx*he.size,Ie=new Uint8Array(H.buffer,he.end*2,ce*2);ee.set(Ie,j),j+=ce*2,he.end+=ce}return new DataView(ee.buffer)}function ne(p){const E=p.array.slice(p.offset.value,p.offset.value+p.size),I=is(E),H=p.inputChannels.length*p.lines*p.columns*p.totalBytes,ie=new ArrayBuffer(H),x=new DataView(ie);let M=0,F=0;const P=new Array(4);for(let R=0;R<p.lines;R++)for(let W=0;W<p.inputChannels.length;W++){let Q=0;switch(p.inputChannels[W].pixelType){case 1:P[0]=M,P[1]=P[0]+p.columns,M=P[1]+p.columns;for(let ee=0;ee<p.columns;++ee){const ae=I[P[0]++]<<8|I[P[1]++];Q+=ae,x.setUint16(F,Q,!0),F+=2}break;case 2:P[0]=M,P[1]=P[0]+p.columns,P[2]=P[1]+p.columns,M=P[2]+p.columns;for(let ee=0;ee<p.columns;++ee){const ae=I[P[0]++]<<24|I[P[1]++]<<16|I[P[2]++]<<8;Q+=ae,x.setUint32(F,Q,!0),F+=4}break}}return x}function Ce(p){const E=p.viewer,I={value:p.offset.value},H=new Uint8Array(p.columns*p.lines*(p.inputChannels.length*p.type*2)),ie={version:He(E,I),unknownUncompressedSize:He(E,I),unknownCompressedSize:He(E,I),acCompressedSize:He(E,I),dcCompressedSize:He(E,I),rleCompressedSize:He(E,I),rleUncompressedSize:He(E,I),rleRawSize:He(E,I),totalAcUncompressedCount:He(E,I),totalDcUncompressedCount:He(E,I),acCompression:He(E,I)};if(ie.version<2)throw new Error("EXRLoader.parse: "+vn.compression+" version "+ie.version+" is unsupported");const x=new Array;let M=we(E,I)-2;for(;M>0;){const se=xe(E.buffer,I),he=Ee(E,I),ce=he>>2&3,Ie=(he>>4)-1,We=new Int8Array([Ie])[0],je=Ee(E,I);x.push({name:se,index:We,type:je,compression:ce}),M-=se.length+3}const F=vn.channels,P=new Array(p.inputChannels.length);for(let se=0;se<p.inputChannels.length;++se){const he=P[se]={},ce=F[se];he.name=ce.name,he.compression=0,he.decoded=!1,he.type=ce.pixelType,he.pLinear=ce.pLinear,he.width=p.columns,he.height=p.lines}const R={idx:new Array(3)};for(let se=0;se<p.inputChannels.length;++se){const he=P[se];for(let ce=0;ce<x.length;++ce){const Ie=x[ce];he.name==Ie.name&&(he.compression=Ie.compression,Ie.index>=0&&(R.idx[Ie.index]=se),he.offset=se)}}let W,Q,j;if(ie.acCompressedSize>0)switch(ie.acCompression){case 0:W=new Uint16Array(ie.totalAcUncompressedCount),Fe(p.array,E,I,ie.acCompressedSize,W,ie.totalAcUncompressedCount);break;case 1:const se=p.array.slice(I.value,I.value+ie.totalAcUncompressedCount),he=is(se);W=new Uint16Array(he.buffer),I.value+=ie.totalAcUncompressedCount;break}if(ie.dcCompressedSize>0){const se={array:p.array,offset:I,size:ie.dcCompressedSize};Q=new Uint16Array(re(se).buffer),I.value+=ie.dcCompressedSize}if(ie.rleRawSize>0){const se=p.array.slice(I.value,I.value+ie.rleCompressedSize),he=is(se);j=dt(he.buffer),I.value+=ie.rleCompressedSize}let ee=0;const ae=new Array(P.length);for(let se=0;se<ae.length;++se)ae[se]=new Array;for(let se=0;se<p.lines;++se)for(let he=0;he<P.length;++he)ae[he].push(ee),ee+=P[he].width*p.type*2;R.idx[0]!==void 0&&P[R.idx[0]]&&Ke(R,ae,P,W,Q,H);for(let se=0;se<P.length;++se){const he=P[se];if(!he.decoded)switch(he.compression){case 2:let ce=0,Ie=0;for(let We=0;We<p.lines;++We){let je=ae[se][ce];for(let Oe=0;Oe<he.width;++Oe){for(let Ue=0;Ue<2*he.type;++Ue)H[je++]=j[Ie+Ue*he.width*he.height];Ie++}ce++}break;case 1:St(se,ae,P,W,Q,H);break;default:throw new Error("EXRLoader.parse: unsupported channel compression")}}return new DataView(H.buffer)}function xe(p,E){const I=new Uint8Array(p);let H=0;for(;I[E.value+H]!=0;)H+=1;const ie=new TextDecoder().decode(I.slice(E.value,E.value+H));return E.value=E.value+H+1,ie}function ke(p,E,I){const H=new TextDecoder().decode(new Uint8Array(p).slice(E.value,E.value+I));return E.value=E.value+I,H}function Ve(p,E){const I=pe(p,E),H=Te(p,E);return[I,H]}function me(p,E){const I=Te(p,E),H=Te(p,E);return[I,H]}function pe(p,E){const I=p.getInt32(E.value,!0);return E.value=E.value+4,I}function Te(p,E){const I=p.getUint32(E.value,!0);return E.value=E.value+4,I}function Re(p,E){const I=p[E.value];return E.value=E.value+1,I}function Ee(p,E){const I=p.getUint8(E.value);return E.value=E.value+1,I}const He=function(p,E){let I;return"getBigInt64"in DataView.prototype?I=Number(p.getBigInt64(E.value,!0)):I=p.getUint32(E.value+4,!0)+Number(p.getUint32(E.value,!0)<<32),E.value+=8,I};function k(p,E){const I=p.getFloat32(E.value,!0);return E.value+=4,I}function Se(p,E){return Wo.toHalfFloat(k(p,E))}function J(p){const E=(p&31744)>>10,I=p&1023;return(p>>15?-1:1)*(E?E===31?I?NaN:1/0:Math.pow(2,E-15)*(1+I/1024):6103515625e-14*(I/1024))}function we(p,E){const I=p.getUint16(E.value,!0);return E.value+=2,I}function ge(p,E){return J(we(p,E))}function te(p,E,I,H){const ie=I.value,x=[];for(;I.value<ie+H-1;){const M=xe(E,I),F=pe(p,I),P=Ee(p,I);I.value+=3;const R=pe(p,I),W=pe(p,I);x.push({name:M,pixelType:F,pLinear:P,xSampling:R,ySampling:W})}return I.value+=1,x}function Pe(p,E){const I=k(p,E),H=k(p,E),ie=k(p,E),x=k(p,E),M=k(p,E),F=k(p,E),P=k(p,E),R=k(p,E);return{redX:I,redY:H,greenX:ie,greenY:x,blueX:M,blueY:F,whiteX:P,whiteY:R}}function qe(p,E){const I=["NO_COMPRESSION","RLE_COMPRESSION","ZIPS_COMPRESSION","ZIP_COMPRESSION","PIZ_COMPRESSION","PXR24_COMPRESSION","B44_COMPRESSION","B44A_COMPRESSION","DWAA_COMPRESSION","DWAB_COMPRESSION"],H=Ee(p,E);return I[H]}function mt(p,E){const I=pe(p,E),H=pe(p,E),ie=pe(p,E),x=pe(p,E);return{xMin:I,yMin:H,xMax:ie,yMax:x}}function ut(p,E){const I=["INCREASING_Y","DECREASING_Y","RANDOM_Y"],H=Ee(p,E);return I[H]}function ln(p,E){const I=["ENVMAP_LATLONG","ENVMAP_CUBE"],H=Ee(p,E);return I[H]}function cn(p,E){const I=["ONE_LEVEL","MIPMAP_LEVELS","RIPMAP_LEVELS"],H=["ROUND_DOWN","ROUND_UP"],ie=Te(p,E),x=Te(p,E),M=Ee(p,E);return{xSize:ie,ySize:x,levelMode:I[M&15],roundingMode:H[M>>4]}}function Mr(p,E){const I=k(p,E),H=k(p,E);return[I,H]}function $i(p,E){const I=k(p,E),H=k(p,E),ie=k(p,E);return[I,H,ie]}function Ms(p,E,I,H,ie){if(H==="string"||H==="stringvector"||H==="iccProfile")return ke(E,I,ie);if(H==="chlist")return te(p,E,I,ie);if(H==="chromaticities")return Pe(p,I);if(H==="compression")return qe(p,I);if(H==="box2i")return mt(p,I);if(H==="envmap")return ln(p,I);if(H==="tiledesc")return cn(p,I);if(H==="lineOrder")return ut(p,I);if(H==="float")return k(p,I);if(H==="v2f")return Mr(p,I);if(H==="v3f")return $i(p,I);if(H==="int")return pe(p,I);if(H==="rational")return Ve(p,I);if(H==="timecode")return me(p,I);if(H==="preview")return I.value+=ie,"skipped";I.value+=ie}function yr(p,E){const I=Math.log2(p);return E=="ROUND_DOWN"?Math.floor(I):Math.ceil(I)}function Er(p,E,I){let H=0;switch(p.levelMode){case"ONE_LEVEL":H=1;break;case"MIPMAP_LEVELS":H=yr(Math.max(E,I),p.roundingMode)+1;break;case"RIPMAP_LEVELS":throw new Error("THREE.EXRLoader: RIPMAP_LEVELS tiles currently unsupported.")}return H}function xn(p,E,I,H){const ie=new Array(p);for(let x=0;x<p;x++){const M=1<<x;let F=E/M|0;H=="ROUND_UP"&&F*M<E&&(F+=1);const P=Math.max(F,1);ie[x]=(P+I-1)/I|0}return ie}function ji(){const p=this,E=p.offset,I={value:0};for(let H=0;H<p.tileCount;H++){const ie=pe(p.viewer,E),x=pe(p.viewer,E);E.value+=8,p.size=Te(p.viewer,E);const M=ie*p.blockWidth,F=x*p.blockHeight;p.columns=M+p.blockWidth>p.width?p.width-M:p.blockWidth,p.lines=F+p.blockHeight>p.height?p.height-F:p.blockHeight;const P=p.columns*p.totalBytes,W=p.size<p.lines*P?p.uncompress(p):v(p);E.value+=p.size;for(let Q=0;Q<p.lines;Q++){const j=Q*p.columns*p.totalBytes;for(let ee=0;ee<p.inputChannels.length;ee++){const ae=vn.channels[ee].name,se=p.channelByteOffsets[ae]*p.columns,he=p.decodeChannels[ae];if(he===void 0)continue;I.value=j+se;const ce=(p.height-(1+F+Q))*p.outLineWidth;for(let Ie=0;Ie<p.columns;Ie++){const We=ce+(Ie+M)*p.outputChannels+he;p.byteArray[We]=p.getter(W,I)}}}}}function br(){const p=this,E=p.offset,I={value:0};for(let H=0;H<p.height/p.blockHeight;H++){const ie=pe(p.viewer,E)-vn.dataWindow.yMin;p.size=Te(p.viewer,E),p.lines=ie+p.blockHeight>p.height?p.height-ie:p.blockHeight;const x=p.columns*p.totalBytes,F=p.size<p.lines*x?p.uncompress(p):v(p);E.value+=p.size;for(let P=0;P<p.blockHeight;P++){const R=H*p.blockHeight,W=P+p.scanOrder(R);if(W>=p.height)continue;const Q=P*x,j=(p.height-1-W)*p.outLineWidth;for(let ee=0;ee<p.inputChannels.length;ee++){const ae=vn.channels[ee].name,se=p.channelByteOffsets[ae]*p.columns,he=p.decodeChannels[ae];if(he!==void 0){I.value=Q+se;for(let ce=0;ce<p.columns;ce++){const Ie=j+ce*p.outputChannels+he;p.byteArray[Ie]=p.getter(F,I)}}}}}}function Tr(p,E,I){const H={};if(p.getUint32(0,!0)!=20000630)throw new Error("THREE.EXRLoader: Provided file doesn't appear to be in OpenEXR format.");H.version=p.getUint8(4);const ie=p.getUint8(5);H.spec={singleTile:!!(ie&2),longName:!!(ie&4),deepFormat:!!(ie&8),multiPart:!!(ie&16)},I.value=8;let x=!0;for(;x;){const M=xe(E,I);if(M==="")x=!1;else{const F=xe(E,I),P=Te(p,I),R=Ms(p,E,I,F,P);R===void 0?console.warn(`THREE.EXRLoader: Skipped unknown header attribute type '${F}'.`):H[M]=R}}if((ie&-7)!=0)throw console.error("THREE.EXRHeader:",H),new Error("THREE.EXRLoader: Provided file is currently unsupported.");return H}function Si(p,E,I,H,ie,x){const M={size:0,viewer:E,array:I,offset:H,width:p.dataWindow.xMax-p.dataWindow.xMin+1,height:p.dataWindow.yMax-p.dataWindow.yMin+1,inputChannels:p.channels,channelByteOffsets:{},shouldExpand:!1,scanOrder:null,totalBytes:null,columns:null,lines:null,type:null,uncompress:null,getter:null,format:null,colorSpace:jt};switch(p.compression){case"NO_COMPRESSION":M.blockHeight=1,M.uncompress=v;break;case"RLE_COMPRESSION":M.blockHeight=1,M.uncompress=z;break;case"ZIPS_COMPRESSION":M.blockHeight=1,M.uncompress=re;break;case"ZIP_COMPRESSION":M.blockHeight=16,M.uncompress=re;break;case"PIZ_COMPRESSION":M.blockHeight=32,M.uncompress=le;break;case"PXR24_COMPRESSION":M.blockHeight=16,M.uncompress=ne;break;case"DWAA_COMPRESSION":M.blockHeight=32,M.uncompress=Ce;break;case"DWAB_COMPRESSION":M.blockHeight=256,M.uncompress=Ce;break;default:throw new Error("EXRLoader.parse: "+p.compression+" is unsupported")}const F={};for(const j of p.channels)switch(j.name){case"Y":case"R":case"G":case"B":case"A":F[j.name]=!0,M.type=j.pixelType}let P=!1,R=!1;if(F.R&&F.G&&F.B)M.outputChannels=4;else if(F.Y)M.outputChannels=1;else throw new Error("EXRLoader.parse: file contains unsupported data channels.");switch(M.outputChannels){case 4:x==Ft?(P=!F.A,M.format=Ft,M.colorSpace=jt,M.outputChannels=4,M.decodeChannels={R:0,G:1,B:2,A:3}):x==mn?(M.format=mn,M.colorSpace=jt,M.outputChannels=2,M.decodeChannels={R:0,G:1}):x==gi?(M.format=gi,M.colorSpace=jt,M.outputChannels=1,M.decodeChannels={R:0}):R=!0;break;case 1:x==Ft?(P=!0,M.format=Ft,M.colorSpace=jt,M.outputChannels=4,M.shouldExpand=!0,M.decodeChannels={Y:0}):x==mn?(M.format=mn,M.colorSpace=jt,M.outputChannels=2,M.shouldExpand=!0,M.decodeChannels={Y:0}):x==gi?(M.format=gi,M.colorSpace=jt,M.outputChannels=1,M.decodeChannels={Y:0}):R=!0;break;default:R=!0}if(R)throw new Error("EXRLoader.parse: invalid output format for specified file.");if(M.type==1)switch(ie){case Yt:M.getter=ge;break;case Jt:M.getter=we;break}else if(M.type==2)switch(ie){case Yt:M.getter=k;break;case Jt:M.getter=Se}else throw new Error("EXRLoader.parse: unsupported pixelType "+M.type+" for "+p.compression+".");M.columns=M.width;const W=M.width*M.height*M.outputChannels;switch(ie){case Yt:M.byteArray=new Float32Array(W),P&&M.byteArray.fill(1,0,W);break;case Jt:M.byteArray=new Uint16Array(W),P&&M.byteArray.fill(15360,0,W);break;default:console.error("THREE.EXRLoader: unsupported type: ",ie);break}let Q=0;for(const j of p.channels)M.decodeChannels[j.name]!==void 0&&(M.channelByteOffsets[j.name]=Q),Q+=j.pixelType*2;if(M.totalBytes=Q,M.outLineWidth=M.width*M.outputChannels,p.lineOrder==="INCREASING_Y"?M.scanOrder=j=>j:M.scanOrder=j=>M.height-1-j,p.spec.singleTile){M.blockHeight=p.tiles.ySize,M.blockWidth=p.tiles.xSize;const j=Er(p.tiles,M.width,M.height),ee=xn(j,M.width,p.tiles.xSize,p.tiles.roundingMode),ae=xn(j,M.height,p.tiles.ySize,p.tiles.roundingMode);M.tileCount=ee[0]*ae[0];for(let se=0;se<j;se++)for(let he=0;he<ae[se];he++)for(let ce=0;ce<ee[se];ce++)He(E,H);M.decode=ji.bind(M)}else{M.blockWidth=M.width;const j=Math.ceil(M.height/M.blockHeight);for(let ee=0;ee<j;ee++)He(E,H);M.decode=br.bind(M)}return M}const Ki={value:0},ai=new DataView(e),Ar=new Uint8Array(e),vn=Tr(ai,e,Ki),Sn=Si(vn,ai,Ar,Ki,this.type,this.outputFormat);if(Sn.decode(),Sn.shouldExpand){const p=Sn.byteArray;if(this.outputFormat==Ft)for(let E=0;E<p.length;E+=4)p[E+2]=p[E+1]=p[E];else if(this.outputFormat==mn)for(let E=0;E<p.length;E+=2)p[E+1]=p[E]}return{header:vn,width:Sn.width,height:Sn.height,data:Sn.byteArray,format:Sn.format,colorSpace:Sn.colorSpace,type:this.type}}setDataType(e){return this.type=e,this}setOutputFormat(e){return this.outputFormat=e,this}load(e,t,n,r){function s(a,o){a.colorSpace=o.colorSpace,a.minFilter=bt,a.magFilter=bt,a.generateMipmaps=!1,a.flipY=!1,t&&t(a,o)}return super.load(e,s,n,r)}}class Vg{constructor(e){this.scene=new wh,new Hg().load("./resources/meadow_4k.exr",n=>{n.mapping=ss,this.scene.background=n}),this.camera=new rn(75,window.innerWidth/window.innerHeight,1,2e4),this.renderer=new ag({antialias:!0}),this.renderer.setSize(window.innerWidth,window.innerHeight),this.renderer.setPixelRatio(window.devicePixelRatio),document.getElementById(e).appendChild(this.renderer.domElement),this.controls=new pg(this.camera,this.renderer.domElement),this.controls.enableDamping=!0,this.controls.dampingFactor=.05,this.scene.add(new au(16777215,1),new ou(16777215,.3)),this.boidInstancedMesh=null,this.boundsLine=null,window.addEventListener("resize",this.onWindowResize.bind(this))}async init(){this.renderer&&typeof this.renderer.init=="function"&&await this.renderer.init()}onWindowResize(){this.camera.aspect=window.innerWidth/window.innerHeight,this.camera.updateProjectionMatrix(),this.renderer.setSize(window.innerWidth,window.innerHeight)}updateVisualBounds(e){this.boundsLine&&this.scene.remove(this.boundsLine);const t=new ii(e.x,e.y,e.z),n=new Wh(t);if(this.boundsLine=new Hh(n,new lc({color:4473924})),this.boundsLine.name="boid-bounds",this.boundsLine.position.set(e.x/2,e.y/2,e.z/2),this.scene.add(this.boundsLine),this.controls){this.controls.target.set(e.x/2,e.y/2,e.z/2);const r=e.x/2,s=e.y/2,a=e.z/2,o=e.x;this.camera.position.set(r+o,s+o,a+o)}}createInstancedMesh(e){this.boidInstancedMesh&&(this.scene.remove(this.boidInstancedMesh),this.boidInstancedMesh.geometry.dispose(),this.boidInstancedMesh.material.dispose());const t=new vo(2,6,5).rotateX(Math.PI/2),n=new ii(6,.1,6).translate(3,0,0),r=new ii(6,.1,6).translate(-3,0,0),s=t.attributes.position.count,a=n.attributes.position.count;t.setAttribute("isWing",new It(new Float32Array(s).fill(0),1)),n.setAttribute("isWing",new It(new Float32Array(a).fill(1),1)),r.setAttribute("isWing",new It(new Float32Array(a).fill(1),1));const o=dg([t,n,r],!1),c=new jh({color:16777215});c.onBeforeCompile=h=>{h.uniforms.time={value:0},h.fragmentShader=h.fragmentShader.replace("#include <common>",`
          #include <common>
          varying vec3 vInstanceColor;
        `).replace("#include <color_fragment>",`
          #include <color_fragment>
          diffuseColor.rgb *= vInstanceColor;
        `),h.vertexShader=`
      varying vec3 vInstanceColor;
      ${h.vertexShader}
    `.replace("#include <begin_vertex>",`
      #include <begin_vertex>
      vInstanceColor = instanceColor;
      `),h.vertexShader=`
    attribute float isWing;
    uniform float time;

    // Helper to get a random float from an ID
    float hash(float n) {
        return fract(sin(n) * 43758.5453123);
    }

    ${h.vertexShader}
  `.replace("#include <begin_vertex>",`
      #include <begin_vertex>
      
      if (isWing > 0.5) {
        // Create unique variations per instance
        float id = float(gl_InstanceID);
        float speedVariation = 0.5 + hash(id + 1.0) * 0.5; // Speed between 0.5x and 1.0x
        float phaseOffset = hash(id) * 6.28; // Phase offset between 0 and 2*PI
        
        float phase = (time * 8.0 * speedVariation) + phaseOffset;
        
        float distFromHinge = max(0.0, abs(position.x) - 1.0);
        float angle = sin(phase) * 0.5;
        
        transformed.y += distFromHinge * angle;
        transformed.z += distFromHinge * abs(angle) * 0.2;
      }
    `),c.userData.shaderUniforms=h.uniforms},this.boidInstancedMesh=new Oh(o,c,e);const l=new rt(1050884),u=new rt(11184810),d=new rt;for(let h=0;h<e;h++)d.lerpColors(l,u,Math.random()),this.boidInstancedMesh.setColorAt(h,d);this.boidInstancedMesh.instanceMatrix.setUsage(oh),this.scene.add(this.boidInstancedMesh)}updateInstances(e){this.boidInstancedMesh&&(this.boidInstancedMesh.instanceMatrix.array.set(e),this.boidInstancedMesh.instanceMatrix.needsUpdate=!0)}render(e){if(this.controls&&this.controls.update(),this.boidInstancedMesh&&this.boidInstancedMesh.material&&this.boidInstancedMesh.material.userData&&this.boidInstancedMesh.material.userData.shaderUniforms){const t=this.boidInstancedMesh.material.userData.shaderUniforms;t.time&&(t.time.value=e*.001)}this.renderer.render(this.scene,this.camera)}}const Ut={IDLE:0,WARMING_UP:1,RECORDING:2,COMPLETED:3};class Wg{constructor(e,t=null,n){this.state=Ut.IDLE,this.frameTimes=[],this.lastFrameTime=0,this.onResetCallback=e,this.onCompleteCallback=t,this.reportExporter=n,this.WARM_UP_MS=1e4,this.RECORD_MS=1e4,this.warmUpTimeout=null,this.recordTimeout=null,this.warmUpEndsAt=0,this.recordEndsAt=0,this.simFrameSamples=[],this.renderFrameSamples=[],this.onEscHandler=null}registerCancelHotkey(){typeof window>"u"||this.onEscHandler||(this.onEscHandler=e=>{e.key==="Escape"&&(this.state!==Ut.WARMING_UP&&this.state!==Ut.RECORDING||(e.preventDefault(),this.cancelBenchmark("Benchmark canceled by user (Esc).")))},window.addEventListener("keydown",this.onEscHandler))}unregisterCancelHotkey(){typeof window>"u"||!this.onEscHandler||(window.removeEventListener("keydown",this.onEscHandler),this.onEscHandler=null)}finalizeRun(){this.warmUpTimeout&&(clearTimeout(this.warmUpTimeout),this.warmUpTimeout=null),this.recordTimeout&&(clearTimeout(this.recordTimeout),this.recordTimeout=null),this.unregisterCancelHotkey(),this.state=Ut.IDLE,this.warmUpEndsAt=0,this.recordEndsAt=0,this.onCompleteCallback&&this.onCompleteCallback()}cancelBenchmark(e="Benchmark canceled."){this.state!==Ut.WARMING_UP&&this.state!==Ut.RECORDING||(this.finalizeRun(),console.log(e))}start(){if(this.state!==Ut.IDLE&&this.state!==Ut.COMPLETED){console.warn("Benchmark already in progress.");return}this.frameTimes=[],this.simFrameSamples=[],this.renderFrameSamples=[],this.lastFrameTime=0,this.state=Ut.WARMING_UP,this.warmUpEndsAt=performance.now()+this.WARM_UP_MS,this.recordEndsAt=0,this.registerCancelHotkey(),this.onResetCallback(),console.log("Benchmark: WARMING UP (10s)..."),this.warmUpTimeout=setTimeout(()=>{this.state=Ut.RECORDING,this.lastFrameTime=performance.now(),this.recordEndsAt=performance.now()+this.RECORD_MS,console.log("Benchmark: RECORDING (10s)..."),this.recordTimeout=setTimeout(()=>{this.completeBenchmark()},this.RECORD_MS)},this.WARM_UP_MS)}recordFrame(e=null,t=null){const n=typeof performance<"u"&&typeof performance.now=="function"?performance.now():Date.now(),r=Number.isFinite(t)?t:Number.isFinite(e)?e:n;if(this.state!==Ut.RECORDING){this.lastFrameTime=r;return}const s=r-this.lastFrameTime;s>0&&this.frameTimes.push(s),this.lastFrameTime=r}recordSimulationSample(e){this.state===Ut.RECORDING&&Number.isFinite(e)&&this.simFrameSamples.push(e)}recordRenderSample(e){this.state===Ut.RECORDING&&Number.isFinite(e)&&this.renderFrameSamples.push(e)}getStatus(e=performance.now()){return this.state===Ut.WARMING_UP?{visible:!0,phaseClass:"warming",status:"Warming Up",detail:`${Math.max(0,(this.warmUpEndsAt-e)/1e3).toFixed(1)}s remaining`}:this.state===Ut.RECORDING?{visible:!0,phaseClass:"recording",status:"Recording",detail:`${Math.max(0,(this.recordEndsAt-e)/1e3).toFixed(1)}s remaining`}:this.state===Ut.COMPLETED?{visible:!0,phaseClass:"completed",status:"Completed",detail:"Preparing export..."}:{visible:!1,phaseClass:"",status:"Idle",detail:""}}async completeBenchmark(e){if(this.state=Ut.COMPLETED,this.unregisterCancelHotkey(),console.log(`Benchmark COMPLETED. Captured ${this.frameTimes.length} frames.`),this.frameTimes.length===0){console.warn("Benchmark completed without captured frame times."),this.finalizeRun();return}const t=n=>n.length?n.reduce((r,s)=>r+s,0)/n.length:0;try{await this.reportExporter.exportPerformanceReport({frameTimes:this.frameTimes,settings:e,hardware:{cpu:"",gpu:"",os:""},metrics:{avgRenderTime:t(this.renderFrameSamples),avgSimTime:t(this.simFrameSamples)}})}finally{this.finalizeRun(),console.log("Benchmark flow finished. Ready for next run.")}}}class Xg{constructor(e){this.callbacks=e,this.isSimulationRunning=!0,this.wasSettingsPanelOpenBeforeBenchmark=null,this.boidCountInput=document.getElementById("boid-count"),this.boidDensityInput=document.getElementById("boid-density"),this.boidCountUpdateTimer=null}init(e){this.populateInputs(e),this.setupEventListeners(),this.updateStartPauseButton(),this.initTooltips()}populateInputs(e){const t=n=>parseFloat(n.toPrecision(6));document.getElementById("boid-count").value=e.boidCount,document.getElementById("boid-density").value=e.boidDensity.toFixed(6),document.getElementById("separation").value=t(e.params[Be.SEPARATION_DIST]),document.getElementById("align").value=t(e.params[Be.ALIGN_DIST]),document.getElementById("cohesion").value=t(e.params[Be.COHESION_DIST]),document.getElementById("max_speed").value=t(e.params[Be.MAX_SPEED]),document.getElementById("max_force").value=t(e.params[Be.MAX_FORCE]),document.getElementById("sep_weight").value=t(e.params[Be.SEPARATION_WEIGHT]),document.getElementById("align_weight").value=t(e.params[Be.ALIGNMENT_WEIGHT]),document.getElementById("coh_weight").value=t(e.params[Be.COHESION_WEIGHT]),document.getElementById("margin").value=t(e.params[Be.MARGIN]),document.getElementById("turn_factor").value=t(e.params[Be.TURN_FACTOR]),document.getElementById("vision_angle").value=t(e.params[Be.VISION_ANGLE]*180/Math.PI)}setupEventListeners(){const e=()=>{const n=parseInt(this.boidCountInput.value,10);!isNaN(n)&&n>0&&n!==this.callbacks.getBoidCount()&&this.callbacks.onRecreateBoids(n,parseFloat(this.boidDensityInput.value))};this.boidCountInput.addEventListener("change",e),this.boidCountInput.addEventListener("input",()=>{this.boidCountUpdateTimer&&clearTimeout(this.boidCountUpdateTimer),this.boidCountUpdateTimer=setTimeout(e,250)}),this.boidDensityInput.addEventListener("input",n=>{const r=parseFloat(n.target.value);!isNaN(r)&&r>0&&this.callbacks.onRecreateBoids(parseInt(this.boidCountInput.value,10),r)}),["separation","align","cohesion","max_speed","max_force","sep_weight","align_weight","coh_weight","margin","turn_factor","vision_angle"].forEach(n=>{document.getElementById(n).addEventListener("input",()=>{this.callbacks.onUpdateUniforms(this.getUniformValues())})}),document.getElementById("toggle-panel").addEventListener("click",()=>{const n=document.getElementById("settings-body");bootstrap.Collapse.getOrCreateInstance(n).toggle()}),document.getElementById("start-pause-btn").addEventListener("click",()=>{this.isSimulationRunning=!this.isSimulationRunning,this.updateStartPauseButton(),this.callbacks.onSimulationToggle(this.isSimulationRunning)}),document.getElementById("restart-btn").addEventListener("click",()=>{const n=parseInt(this.boidCountInput.value,10);let r=parseFloat(this.boidDensityInput.value);(isNaN(r)||r<=0)&&(r=this.callbacks.getBoidDensity()),(isNaN(n)||n<=0)&&(n=this.callbacks.getBoidCount()),this.callbacks.onRecreateBoids(n,r),this.isSimulationRunning=!0,this.updateStartPauseButton(),this.callbacks.onSimulationToggle(!0)}),document.getElementById("reset-btn").addEventListener("click",()=>{this.callbacks.onResetSimulation()}),document.getElementById("benchmark-btn").addEventListener("click",()=>{this.callbacks.onBenchmarkStart()}),document.getElementById("import-report-btn").addEventListener("click",()=>{const n=document.getElementById("benchmark-json-input");n&&n.click()}),document.getElementById("benchmark-json-input").addEventListener("change",async n=>{const r=n.target.files?Array.from(n.target.files):[];if(r.length!==0)try{await this.callbacks.onImportReport(r)}catch(s){console.error("Failed to import benchmark TeX:",s),window.alert("Failed to import benchmark TeX. See console for details.")}finally{n.target.value=""}})}getUniformValues(){return{separation:parseFloat(document.getElementById("separation").value),align:parseFloat(document.getElementById("align").value),cohesion:parseFloat(document.getElementById("cohesion").value),max_speed:parseFloat(document.getElementById("max_speed").value),max_force:parseFloat(document.getElementById("max_force").value),sep_weight:parseFloat(document.getElementById("sep_weight").value),align_weight:parseFloat(document.getElementById("align_weight").value),coh_weight:parseFloat(document.getElementById("coh_weight").value),margin:parseFloat(document.getElementById("margin").value),turn_factor:parseFloat(document.getElementById("turn_factor").value),vision_angle:parseFloat(document.getElementById("vision_angle").value)}}updateStartPauseButton(){const e=document.getElementById("start-pause-btn"),t=document.getElementById("start-icon");this.isSimulationRunning?(t.className="bi bi-pause-fill",e.classList.add("btn-success"),e.classList.remove("btn-warning")):(t.className="bi bi-play-fill",e.classList.add("btn-warning"),e.classList.remove("btn-success"))}initTooltips(){document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(e=>{bootstrap.Tooltip.getOrCreateInstance(e,{trigger:"hover"})})}updateFPS(e,t,n){document.getElementById("info-fps").innerText=`FPS: ${e.toFixed(1)}`,t!==null&&(document.getElementById("info-step").innerText=`Sim: ${t.toFixed(2)} ms`),n!==null&&(document.getElementById("info-gpu").innerText=`Render: ${n.toFixed(2)} ms`)}updateInfo(e,t){document.getElementById("info-boids").innerText=`Boids: ${e}`,document.getElementById("gpu-status").innerText=`Cells: ${t}`}collapseSettingsPanelForBenchmark(){const e=document.getElementById("settings-body");if(!e)return;this.wasSettingsPanelOpenBeforeBenchmark=e.classList.contains("show"),bootstrap.Collapse.getOrCreateInstance(e,{toggle:!1}).hide()}restoreSettingsPanelAfterBenchmark(){if(this.wasSettingsPanelOpenBeforeBenchmark===null)return;const e=document.getElementById("settings-body");if(!e){this.wasSettingsPanelOpenBeforeBenchmark=null;return}const t=bootstrap.Collapse.getOrCreateInstance(e,{toggle:!1});this.wasSettingsPanelOpenBeforeBenchmark?t.show():t.hide(),this.wasSettingsPanelOpenBeforeBenchmark=null}updateBenchmarkHUD(e){const t=document.getElementById("benchmark-hud");if(!t)return;const n=t.querySelector(".status"),r=t.querySelector(".countdown");if(!(!n||!r)){if(t.classList.remove("warming","recording","completed"),!e.visible){t.classList.remove("show");return}t.classList.add("show"),e.phaseClass&&t.classList.add(e.phaseClass),n.textContent=e.status,r.textContent=e.detail}}}const Xt=class Xt{constructor(e){this.exportFormat="graph",this.previewFormat="markdown",this.setOutputFormats(e)}setOutputFormats(e){e&&(e.exportFormat&&this.isExportOutputFormat(e.exportFormat)&&(this.exportFormat=e.exportFormat),e.previewFormat&&this.isPreviewOutputFormat(e.previewFormat)&&(this.previewFormat=e.previewFormat))}getExportFormat(){return this.exportFormat}getPreviewFormat(){return this.previewFormat}isExportOutputFormat(e){return e==="graph"||e==="latex"}isPreviewOutputFormat(e){return e==="markdown"||e==="latex"||e==="graph"}async exportPerformanceReport(e){const t=e.frameTimes.filter(f=>Number.isFinite(f)&&f>0);if(t.length===0)return window.alert("No frame-time data available to export."),null;const n=new Date,r=this.formatDateStamp(n),s=this.formatBoidLabel(e.settings.boidCount),a=await this.promptBenchmarkNameWithPreview(r,s,e.hardware.cpu,e.hardware.gpu);if(!a)return null;const o=a.benchmarkName,c=`${r}_${s}_Boids_${o}`,l=this.computeStats(t);let u=null;const d={...e,hardware:{cpu:a.cpuType,gpu:a.gpuType,os:""}},h=this.buildBenchmarkExportPayload(t,d,l,n,c);if(a.action==="copy"){const f=this.exportFormat==="latex"?this.generateLatexTableForPayload(h,o):this.generateMarkdownReportForPayload(h,o);return this.openClipboardTextModal(f,this.exportFormat==="latex"?"LaTeX Output":"Markdown Output"),null}if(this.exportFormat==="latex"){const f=this.generateLatexTableForPayload(h,o);u=this.downloadTextAsBlob(f,`${c}.tex`,"application/x-tex")}else{const f=`${c}.png`,_=this.renderReportCanvas(t,e,l,n);u=await this.canvasToPngBlob(_),this.downloadBlob(u,f)}return a.exportJson&&this.downloadJson(h,`${c}.json`),u}async importBenchmarkJsonAndExport(e){const t=await e.text();let n;try{n=JSON.parse(t)}catch{return window.alert("Invalid JSON file."),null}const r=this.parseBenchmarkPayload(n);if(!r)return window.alert("JSON does not match expected benchmark export format."),null;const s=r.frameTimes.filter(b=>Number.isFinite(b)&&b>0);if(s.length===0)return window.alert("Imported JSON has no valid frame-time data."),null;const a=r.generatedAtIso?new Date(r.generatedAtIso):new Date,o=Number.isFinite(a.getTime())?a:new Date,c=this.computeStats(s),l={frameTimes:s,settings:r.settings,hardware:r.hardware,metrics:r.metrics},u=this.formatDateStamp(new Date),d=this.formatBoidLabel(r.settings.boidCount),h=await this.promptBenchmarkNameWithPreview(u,d,l.hardware.cpu,l.hardware.gpu);if(!h)return null;const f=h.benchmarkName,_={cpu:h.cpuType,gpu:h.gpuType,os:""};if(r.hardware=_,l.hardware=_,h.action==="copy"){const b=this.exportFormat==="latex"?this.generateLatexTableForPayload(r,f):this.generateMarkdownReportForPayload(r,f);return this.openClipboardTextModal(b,this.exportFormat==="latex"?"LaTeX Output":"Markdown Output"),null}if(this.exportFormat==="latex"){const b=`${u}_${d}_Boids_${f}`,w=this.generateLatexTableForPayload(r,f);return this.downloadTextAsBlob(w,`${b}.tex`,"application/x-tex")}const y=`${u}_${d}_Boids_${f}.png`,g=this.renderReportCanvas(s,l,c,o),m=await this.canvasToPngBlob(g);return this.downloadBlob(m,y),m}async openBenchmarkPreviewFromTexFiles(e){if(e.length===0)return;if(e.length===1){const s=await e[0].text();if(!s.trim()){window.alert("Selected TeX file is empty.");return}this.openClipboardTextModal(s,"LaTeX Output");return}const t=[],n=[];for(const s of e)try{const a=await s.text(),o=this.parseLatexMetricRows(a);if(!o||o.size===0){n.push(s.name);continue}t.push({fileName:s.name,metrics:o})}catch{n.push(s.name)}if(t.length===0){window.alert("No valid benchmark TeX files were selected.");return}const r=this.generateLatexComparisonFromMetricMaps(t);this.openClipboardTextModal(r,"LaTeX Comparison"),n.length>0&&window.alert(`Skipped ${n.length} invalid TeX file(s): ${n.join(", ")}`)}parseLatexMetricRows(e){const t=e.split(/\r?\n/),n=new Map,r=/^(.*?)\s*&\s*(.*?)\s*\\\\\s*$/;for(const s of t){const o=s.trim().match(r);if(!o)continue;const c=this.unescapeLatex(o[1].trim()),l=this.unescapeLatex(o[2].trim());!c||c==="Metric"||n.set(c,l)}return n.size>0?n:null}generateLatexComparisonFromMetricMaps(e){const t=["Benchmark","Boid Count","Separation Weight","Alignment Weight","Cohesion Weight","Max Speed","Update Frequency","Avg Simulation Time (ms)","Avg Render Time (ms)","Avg Frame Time (ms)","Avg FPS","1% Low Frame Time (ms)","1% Low FPS","CPU","GPU"],n=new Set;e.forEach(u=>{u.metrics.forEach((d,h)=>{t.includes(h)||n.add(h)})});const r=[...t,...Array.from(n)],s=" \\\\",a=["Metric",...e.map(u=>u.fileName.replace(/\.tex$/i,""))],o=`l${"p{0.22\\linewidth}".repeat(e.length)}`,c=`${a.map(u=>this.escapeLatexForWrappedCell(u)).join(" & ")}${s}`,l=r.map(u=>{const d=e.map(h=>h.metrics.get(u)||"-");return[u,...d].map(h=>this.escapeLatexForWrappedCell(h)).join(" & ")+s});return["\\begin{table}[htbp]","\\centering","\\rowcolors{2}{gray!15}{white}",`\\begin{tabular}{${o}}`,"\\hline",c,"\\hline",...l,"\\hline","\\end{tabular}","\\caption{Boid benchmark comparison (imported TeX reports)}","\\label{tab:boid-benchmark-tex-comparison}","\\end{table}",""].join(`
`)}unescapeLatex(e){return e.replace(/\\allowbreak\{\}/g,"").replace(/\\textbackslash\{\}/g,"\\").replace(/\\textasciitilde\{\}/g,"~").replace(/\\textasciicircum\{\}/g,"^").replace(/\\&/g,"&").replace(/\\%/g,"%").replace(/\\\$/g,"$").replace(/\\#/g,"#").replace(/\\_/g,"_").replace(/\\\{/g,"{").replace(/\\\}/g,"}")}escapeLatexForWrappedCell(e){return this.escapeLatex(e).replace(/\\_/g,"\\_\\allowbreak{}").replace(/-/g,"-\\allowbreak{}").replace(/\//g,"/\\allowbreak{}").replace(/\\textbackslash\{\}/g,"\\textbackslash{}\\allowbreak{}")}buildComparisonRows(e){return e.map(t=>({benchmark:t.fileName.replace(/\.json$/i,""),boidCount:t.payload.settings.boidCount,separationWeight:t.payload.settings.separationWeight,alignmentWeight:t.payload.settings.alignmentWeight,cohesionWeight:t.payload.settings.cohesionWeight,maxSpeed:t.payload.settings.maxSpeed,avgSimTime:t.payload.metrics.avgSimTime,avgRenderTime:t.payload.metrics.avgRenderTime,avgFrameTime:t.payload.stats.avgFrameTime,avgFps:t.payload.stats.avgFps,onePercentLowFrameTime:t.payload.stats.onePercentLowFrameTime,onePercentLowFps:t.payload.stats.onePercentLowFps}))}generateMarkdownComparisonTable(e){const t=["Benchmark","Boids","Sep W","Align W","Coh W","Max Speed","Avg Sim (ms)","Avg Render (ms)","Avg Frame (ms)","Avg FPS","1% Low (ms)","1% Low FPS"],n=t.map(()=>"---");return["## Benchmark Comparison","","Table-first output for PR comments/previews.","",...[`| ${t.join(" | ")} |`,`| ${n.join(" | ")} |`,...e.map(s=>`| ${[this.escapeMarkdownCell(s.benchmark),this.formatInteger(s.boidCount),this.formatNumber(s.separationWeight,2),this.formatNumber(s.alignmentWeight,2),this.formatNumber(s.cohesionWeight,2),this.formatNumber(s.maxSpeed,2),this.formatNumber(s.avgSimTime,2),this.formatNumber(s.avgRenderTime,2),this.formatNumber(s.avgFrameTime,2),this.formatNumber(s.avgFps,1),this.formatNumber(s.onePercentLowFrameTime,2),this.formatNumber(s.onePercentLowFps,1)].join(" | ")} |`)],"","Notes:","- Avg values are computed from captured benchmark frame times.","- JSON export remains the canonical historical record.",""].join(`
`)}generateLatexComparisonTable(e){const n=["Benchmark","Boids","Sep W","Align W","Coh W","Max Speed","Avg Sim (ms)","Avg Render (ms)","Avg Frame (ms)","Avg FPS","1\\% Low (ms)","1\\% Low FPS"].join(" & "),r=e.map(s=>[this.escapeLatex(s.benchmark),this.formatInteger(s.boidCount),this.formatNumber(s.separationWeight,2),this.formatNumber(s.alignmentWeight,2),this.formatNumber(s.cohesionWeight,2),this.formatNumber(s.maxSpeed,2),this.formatNumber(s.avgSimTime,2),this.formatNumber(s.avgRenderTime,2),this.formatNumber(s.avgFrameTime,2),this.formatNumber(s.avgFps,1),this.formatNumber(s.onePercentLowFrameTime,2),this.formatNumber(s.onePercentLowFps,1)].join(" & ")+" \\\\").join(`
`);return["\\begin{table}[htbp]","\\centering","\\small","\\rowcolors{2}{gray!15}{white}","\\begin{tabular}{lrrrrrrrrrrr}","\\hline",`${n} \\\\`,"\\hline",r,"\\hline","\\end{tabular}","\\caption{Boid benchmark comparison summary}","\\label{tab:boid-benchmark-comparison}","\\end{table}",""].join(`
`)}generateLatexTableForPayload(e,t){return["\\begin{table}[htbp]","\\centering","\\rowcolors{2}{gray!15}{white}","\\begin{tabular}{l p{0.68\\linewidth}}","\\hline","Metric & Value \\\\","\\hline",[["Benchmark",t],["Boid Count",this.formatInteger(e.settings.boidCount)],["Separation Weight",this.formatNumber(e.settings.separationWeight,2)],["Alignment Weight",this.formatNumber(e.settings.alignmentWeight,2)],["Cohesion Weight",this.formatNumber(e.settings.cohesionWeight,2)],["Max Speed",this.formatNumber(e.settings.maxSpeed,2)],["Update Frequency",this.formatInteger(e.settings.updateFrequency??0)],["Avg Simulation Time (ms)",this.formatNumber(e.metrics.avgSimTime,2)],["Avg Render Time (ms)",this.formatNumber(e.metrics.avgRenderTime,2)],["Avg Frame Time (ms)",this.formatNumber(e.stats.avgFrameTime,2)],["Avg FPS",this.formatNumber(e.stats.avgFps,1)],["1% Low Frame Time (ms)",this.formatNumber(e.stats.onePercentLowFrameTime,2)],["1% Low FPS",this.formatNumber(e.stats.onePercentLowFps,1)],["CPU",e.hardware.cpu],["GPU",e.hardware.gpu]].map(([a,o])=>`${this.escapeLatex(a)} & ${this.escapeLatexForWrappedCell(o)} \\\\`).join(`
`),"\\hline","\\end{tabular}",`\\caption{Boid benchmark report: ${this.escapeLatexForWrappedCell(t)}}`,"\\label{tab:boid-benchmark-report}","\\end{table}",""].join(`
`)}generateMarkdownReportForPayload(e,t){return["## Benchmark Report","","| Metric | Value |","| --- | --- |",...[["Benchmark",t],["Boid Count",this.formatInteger(e.settings.boidCount)],["Separation Weight",this.formatNumber(e.settings.separationWeight,2)],["Alignment Weight",this.formatNumber(e.settings.alignmentWeight,2)],["Cohesion Weight",this.formatNumber(e.settings.cohesionWeight,2)],["Max Speed",this.formatNumber(e.settings.maxSpeed,2)],["Update Frequency",this.formatInteger(e.settings.updateFrequency??0)],["Avg Simulation Time (ms)",this.formatNumber(e.metrics.avgSimTime,2)],["Avg Render Time (ms)",this.formatNumber(e.metrics.avgRenderTime,2)],["Avg Frame Time (ms)",this.formatNumber(e.stats.avgFrameTime,2)],["Avg FPS",this.formatNumber(e.stats.avgFps,1)],["1% Low Frame Time (ms)",this.formatNumber(e.stats.onePercentLowFrameTime,2)],["1% Low FPS",this.formatNumber(e.stats.onePercentLowFps,1)],["CPU",e.hardware.cpu],["GPU",e.hardware.gpu]].map(([s,a])=>`| ${this.escapeMarkdownCell(s)} | ${this.escapeMarkdownCell(a)} |`),""].join(`
`)}formatInteger(e){return Number.isFinite(e)?Math.round(e).toString():"0"}formatNumber(e,t=2){return Number.isFinite(e)?e.toFixed(t):"0"}escapeMarkdownCell(e){return e.replace(/\|/g,"\\|").replace(/\r?\n/g," ")}escapeLatex(e){return e.replace(/\\/g,"\\textbackslash{}").replace(/&/g,"\\&").replace(/%/g,"\\%").replace(/\$/g,"\\$").replace(/#/g,"\\#").replace(/_/g,"\\_").replace(/{/g,"\\{").replace(/}/g,"\\}").replace(/~/g,"\\textasciitilde{}").replace(/\^/g,"\\textasciicircum{}")}renderImportedComparisonCanvas(e){const t=[{label:"Boid Count",unit:"#",values:e.map(_e=>_e.payload.settings.boidCount)},{label:"Separation Weight",unit:"x",values:e.map(_e=>_e.payload.settings.separationWeight)},{label:"Alignment Weight",unit:"x",values:e.map(_e=>_e.payload.settings.alignmentWeight)},{label:"Cohesion Weight",unit:"x",values:e.map(_e=>_e.payload.settings.cohesionWeight)},{label:"Max Speed",unit:"u/s",values:e.map(_e=>_e.payload.settings.maxSpeed)},{label:"Update Frequency",unit:"wg",values:e.map(_e=>_e.payload.settings.updateFrequency??0)},{label:"Avg Sim Time",unit:"ms",values:e.map(_e=>_e.payload.metrics.avgSimTime)},{label:"Avg Render Time",unit:"ms",values:e.map(_e=>_e.payload.metrics.avgRenderTime)},{label:"Avg Frame Time",unit:"ms",values:e.map(_e=>_e.payload.stats.avgFrameTime)},{label:"Avg FPS",unit:"FPS",values:e.map(_e=>_e.payload.stats.avgFps)},{label:"1% Low Frame Time",unit:"ms",values:e.map(_e=>_e.payload.stats.onePercentLowFrameTime)},{label:"1% Low FPS",unit:"FPS",values:e.map(_e=>_e.payload.stats.onePercentLowFps)}],n=80,r=120,s=760,a=130,o=24,c=210,l=2,u=Math.ceil(t.length/l),d=u*c+(u-1)*o,h=document.createElement("canvas");h.width=2400,h.height=n*2+r+s+a+30+d;const f=h.getContext("2d");if(!f)throw new Error("Failed to create 2D canvas context.");const _="#FFFFFF",y="#111111",g="#555555",m="#D9D9D9",b="#D23B3B",w=["#2F80ED","#0AA174","#E07A12","#8A5CF6","#D23B3B","#3FA9F5","#1F7A8C"],A=n,U=n+r,L=h.width-n*2,N=s;f.fillStyle=_,f.fillRect(0,0,h.width,h.height),f.strokeStyle=m,f.lineWidth=3,f.strokeRect(6,6,h.width-12,h.height-12),f.fillStyle=y,f.font="700 52px Arial",f.fillText(e.length>1?"BENCHMARK COMPARISON":"BENCHMARK GRAPH PREVIEW",n,n+52),f.fillStyle=g,f.font="500 24px Arial",f.fillText(e.length>1?"Comparison across all numeric benchmark variables + 60-frame rolling frame-time trend":"Detailed variable view + frame-time trend",n,n+92),f.strokeStyle=m,f.lineWidth=2,f.strokeRect(A,U,L,N);const S=95,T=30,G=25,D=80,O=A+S,V=U+G,K=L-S-T,Y=N-G-D,Z=e.map(_e=>{const $=_e.payload.frameTimes.filter(ue=>Number.isFinite(ue)&&ue>0);return $.length>0?this.computeRollingAverage($,60):[]}),X=Z.flat(),fe=Math.max(1,...Z.map(_e=>_e.length)),oe=Math.max(22,...X,16.67)*1.08,ye=[0,8,16.67,24,33,40,50,66].filter(_e=>_e<=oe+2);f.strokeStyle="#E9E9E9",f.lineWidth=1;for(const _e of ye){const $=V+Y-_e/oe*Y;f.beginPath(),f.moveTo(O,$),f.lineTo(O+K,$),f.stroke(),f.fillStyle=g,f.font="500 19px Arial",f.fillText(`${_e.toFixed(_e===16.67?2:0)}ms`,A+10,$+6)}const Ae=V+Y-16.67/oe*Y;f.strokeStyle=b,f.lineWidth=2,f.setLineDash([10,8]),f.beginPath(),f.moveTo(O,Ae),f.lineTo(O+K,Ae),f.stroke(),f.setLineDash([]),f.fillStyle=b,f.font="600 20px Arial",f.fillText("16.67ms target",O+10,Ae-8);for(let _e=0;_e<Z.length;_e++){const $=Z[_e];if($.length===0)continue;const ue=w[_e%w.length];f.strokeStyle=ue,f.lineWidth=3,f.beginPath();for(let de=0;de<$.length;de++){const ze=fe<=1?0:de/(fe-1),Le=O+ze*K,Fe=V+Y-$[de]/oe*Y;de===0?f.moveTo(Le,Fe):f.lineTo(Le,Fe)}f.stroke()}f.fillStyle=y,f.font="600 24px Arial",f.fillText("Frame Time (ms)",A+14,U+30),f.fillText("Frame Index",O+K-160,U+N-20),f.save(),f.translate(A+26,U+N/2+80),f.rotate(-Math.PI/2),f.fillText("Frame Time (ms)",0,0),f.restore();const ve=U+N+32;f.font="600 20px Arial";for(let _e=0;_e<e.length;_e++){const $=w[_e%w.length],ue=_e%2,de=Math.floor(_e/2),ze=n+ue*((h.width-n*2)/2),Le=ve+de*36,Fe=e[_e].fileName.replace(/\.json$/i,""),xt=Fe.length>58?`${Fe.slice(0,55)}...`:Fe;f.strokeStyle=$,f.lineWidth=4,f.beginPath(),f.moveTo(ze,Le),f.lineTo(ze+34,Le),f.stroke(),f.fillStyle=y,f.fillText(`${_e+1}. ${xt}`,ze+44,Le+7)}const Ge=ve+Math.ceil(e.length/2)*36+34,st=Math.floor((h.width-n*2-o)/2);for(let _e=0;_e<t.length;_e++){const $=_e%l,ue=Math.floor(_e/l),de=n+$*(st+o),ze=Ge+ue*(c+o);this.drawScalarComparisonPanel(f,de,ze,st,c,t[_e].label,t[_e].unit,t[_e].values,w,y,g,m)}return h}drawScalarComparisonPanel(e,t,n,r,s,a,o,c,l,u,d,h){e.strokeStyle=h,e.lineWidth=2,e.strokeRect(t,n,r,s),e.fillStyle=u,e.font="700 24px Arial",e.fillText(`${a} (${o})`,t+14,n+32);const f=44,_=18,y=48,g=36,m=t+f,b=n+y,w=r-f-_,A=s-y-g,U=c.filter(G=>Number.isFinite(G)),L=Math.max(1,...U)*1.1;e.strokeStyle="#EFEFEF",e.lineWidth=1;for(let G=0;G<=4;G++){const D=b+A*G/4;e.beginPath(),e.moveTo(m,D),e.lineTo(m+w,D),e.stroke()}const N=Math.max(1,c.length),S=Math.max(8,Math.floor(w*.015)),T=Math.max(8,Math.floor((w-S*(N-1))/N));for(let G=0;G<c.length;G++){const D=Number.isFinite(c[G])?c[G]:0,O=L<=0?0:D/L*A,V=m+G*(T+S),K=b+A-O,Y=l[G%l.length];e.fillStyle=Y,e.fillRect(V,K,T,O),e.fillStyle=d,e.font="600 14px Arial",e.fillText(String(G+1),V+Math.max(0,T/2-4),b+A+16),e.fillStyle=u,e.font="500 13px Arial";const Z=D>=1e3?D.toFixed(0):D.toFixed(2);e.fillText(Z,V,Math.max(n+46,K-4))}}openGraphPreviewModal(e,t,n){const r=document.createElement("div");r.style.position="fixed",r.style.inset="0",r.style.background="rgba(0,0,0,0.45)",r.style.display="flex",r.style.alignItems="center",r.style.justifyContent="center",r.style.zIndex="10010";const s=document.createElement("div");s.style.width="min(94vw, 1320px)",s.style.maxHeight="90vh",s.style.background="#FFFFFF",s.style.border="1px solid #DADADA",s.style.borderRadius="12px",s.style.boxShadow="0 20px 60px rgba(0,0,0,0.25)",s.style.padding="16px",s.style.display="flex",s.style.flexDirection="column",s.style.gap="12px",s.style.fontFamily="Arial, sans-serif";const a=document.createElement("div");a.style.fontSize="22px",a.style.fontWeight="700",a.textContent=n>1?"Benchmark Comparison Preview":"Benchmark Preview";const o=document.createElement("div");o.style.fontSize="14px",o.style.color="#555",o.textContent="Showing 60-frame rolling average; use Export PNG to save this preview.";const c=document.createElement("div");c.style.overflow="auto",c.style.border="1px solid #E5E5E5",c.style.borderRadius="8px",c.style.background="#FAFAFA",e.style.width="100%",e.style.height="auto",e.style.display="block",c.appendChild(e);const l=document.createElement("div");l.style.display="flex",l.style.justifyContent="flex-end",l.style.gap="10px";const u=document.createElement("button");u.textContent="Close",u.style.padding="8px 12px",u.style.border="1px solid #CCC",u.style.background="#FFF",u.style.borderRadius="8px",u.style.cursor="pointer";const d=document.createElement("button");d.textContent="Export PNG",d.style.padding="8px 12px",d.style.border="none",d.style.background="#2F80ED",d.style.color="#FFF",d.style.borderRadius="8px",d.style.cursor="pointer";const h=()=>{window.removeEventListener("keydown",f),r.remove()},f=_=>{_.key==="Escape"&&h()};u.addEventListener("click",h),d.addEventListener("click",async()=>{const _=await this.canvasToPngBlob(e),y=this.sanitizeFilePart(t.replace(/\.png$/i,""))||"Benchmark_Preview";this.downloadBlob(_,`${y}.png`)}),r.addEventListener("click",_=>{_.target===r&&h()}),window.addEventListener("keydown",f),l.appendChild(u),l.appendChild(d),s.appendChild(a),s.appendChild(o),s.appendChild(c),s.appendChild(l),r.appendChild(s),document.body.appendChild(r)}openTablePreviewModal(e,t,n,r){const s=document.createElement("div");s.style.position="fixed",s.style.inset="0",s.style.background="rgba(0,0,0,0.45)",s.style.display="flex",s.style.alignItems="center",s.style.justifyContent="center",s.style.zIndex="10010";const a=document.createElement("div");a.style.width="min(94vw, 1320px)",a.style.maxHeight="90vh",a.style.background="#FFFFFF",a.style.border="1px solid #DADADA",a.style.borderRadius="12px",a.style.boxShadow="0 20px 60px rgba(0,0,0,0.25)",a.style.padding="16px",a.style.display="flex",a.style.flexDirection="column",a.style.gap="12px",a.style.fontFamily="Arial, sans-serif";const o=document.createElement("div");o.style.fontSize="22px",o.style.fontWeight="700",o.textContent=r>1?`${n==="markdown"?"Markdown":"LaTeX"} Comparison Table`:`${n==="markdown"?"Markdown":"LaTeX"} Benchmark Table`;const c=document.createElement("div");c.style.fontSize="14px",c.style.color="#555",c.textContent=n==="markdown"?"PR-ready table output. Copy or export as .md.":"LaTeX-ready table output with escaped special characters.";const l=document.createElement("div");l.style.overflow="auto",l.style.border="1px solid #E5E5E5",l.style.borderRadius="8px",l.style.background="#FAFAFA",l.style.padding="10px";const u=document.createElement("pre");u.style.margin="0",u.style.fontSize="13px",u.style.lineHeight="1.45",u.style.fontFamily="Consolas, 'Courier New', monospace",u.style.whiteSpace="pre",u.textContent=e,l.appendChild(u);const d=document.createElement("div");d.style.display="flex",d.style.justifyContent="flex-end",d.style.gap="10px";const h=document.createElement("button");h.textContent="Close",h.style.padding="8px 12px",h.style.border="1px solid #CCC",h.style.background="#FFF",h.style.borderRadius="8px",h.style.cursor="pointer";const f=document.createElement("button");f.textContent=n==="markdown"?"Copy Markdown":"Copy LaTeX",f.style.padding="8px 12px",f.style.border="none",f.style.background="#0AA174",f.style.color="#FFF",f.style.borderRadius="8px",f.style.cursor="pointer";const _=document.createElement("button");_.textContent=n==="markdown"?"Export .md":"Export .tex",_.style.padding="8px 12px",_.style.border="none",_.style.background="#2F80ED",_.style.color="#FFF",_.style.borderRadius="8px",_.style.cursor="pointer";const y=()=>{window.removeEventListener("keydown",g),s.remove()},g=m=>{m.key==="Escape"&&y()};h.addEventListener("click",y),f.addEventListener("click",async()=>{try{await navigator.clipboard.writeText(e)}catch{window.alert("Copy failed. Clipboard access may be blocked by the browser.")}}),_.addEventListener("click",()=>{const m=n==="markdown"?"text/markdown":"application/x-tex";this.downloadTextAsBlob(e,t,m)}),s.addEventListener("click",m=>{m.target===s&&y()}),window.addEventListener("keydown",g),d.appendChild(h),d.appendChild(f),d.appendChild(_),a.appendChild(o),a.appendChild(c),a.appendChild(l),a.appendChild(d),s.appendChild(a),document.body.appendChild(s)}openClipboardTextModal(e,t){const n=document.createElement("div");n.style.position="fixed",n.style.inset="0",n.style.background="rgba(0,0,0,0.45)",n.style.display="flex",n.style.alignItems="center",n.style.justifyContent="center",n.style.zIndex="10010";const r=document.createElement("div");r.style.width="min(94vw, 980px)",r.style.maxHeight="90vh",r.style.background="#FFFFFF",r.style.border="1px solid #DADADA",r.style.borderRadius="12px",r.style.boxShadow="0 20px 60px rgba(0,0,0,0.25)",r.style.padding="16px",r.style.display="flex",r.style.flexDirection="column",r.style.gap="12px",r.style.fontFamily="Arial, sans-serif";const s=document.createElement("div");s.style.fontSize="22px",s.style.fontWeight="700",s.textContent=`${t} Clipboard`;const a=document.createElement("div");a.style.fontSize="14px",a.style.color="#555",a.textContent="The text below is what will be copied to your clipboard.";const o=t.toLowerCase().includes("latex");let c=null;o&&(c=document.createElement("div"),c.style.fontSize="13px",c.style.background="#EEF6FF",c.style.border="1px solid #B8D8F8",c.style.borderRadius="8px",c.style.padding="10px 14px",c.style.color="#1A4A7A",c.innerHTML='<strong>Required preamble:</strong> add the following to your LaTeX document header before using this table:<br><code style="font-size:12px;background:#DDEEFF;padding:2px 6px;border-radius:4px;">\\usepackage[table]{xcolor}</code>');const l=document.createElement("textarea");l.readOnly=!0,l.value=e,l.style.width="100%",l.style.minHeight="360px",l.style.resize="vertical",l.style.border="1px solid #E5E5E5",l.style.borderRadius="8px",l.style.padding="10px",l.style.fontSize="13px",l.style.lineHeight="1.45",l.style.fontFamily="Consolas, 'Courier New', monospace",l.style.background="#FAFAFA";const u=document.createElement("div");u.style.display="flex",u.style.justifyContent="flex-end",u.style.gap="10px";const d=document.createElement("button");d.textContent="Close",d.style.padding="8px 12px",d.style.border="1px solid #CCC",d.style.background="#FFF",d.style.borderRadius="8px",d.style.cursor="pointer";const h=document.createElement("button");h.textContent="Copy",h.style.padding="8px 12px",h.style.border="none",h.style.background="#0AA174",h.style.color="#FFF",h.style.borderRadius="8px",h.style.cursor="pointer";const f=()=>{window.removeEventListener("keydown",_),n.remove()},_=y=>{y.key==="Escape"&&f()};d.addEventListener("click",f),h.addEventListener("click",async()=>{try{await navigator.clipboard.writeText(e),h.textContent="Copied"}catch{window.alert("Copy failed. Clipboard access may be blocked by the browser.")}}),n.addEventListener("click",y=>{y.target===n&&f()}),window.addEventListener("keydown",_),u.appendChild(d),u.appendChild(h),r.appendChild(s),r.appendChild(a),c&&r.appendChild(c),r.appendChild(l),r.appendChild(u),n.appendChild(r),document.body.appendChild(n),l.focus()}parseBenchmarkPayload(e){if(!e||typeof e!="object")return null;const t=e;if(!Array.isArray(t.frameTimes))return null;const n=t.frameTimes.filter(c=>typeof c=="number");if(n.length===0)return null;const r=t.settings,s=t.hardware,a=t.metrics;if(!r||!s||!a||typeof r.boidCount!="number"||typeof r.separationWeight!="number"||typeof r.alignmentWeight!="number"||typeof r.cohesionWeight!="number"||typeof r.maxSpeed!="number"||typeof s.cpu!="string"||typeof s.gpu!="string"||typeof s.os!="string"||typeof a.avgRenderTime!="number"||typeof a.avgSimTime!="number")return null;const o=this.computeStats(n);return{schemaVersion:typeof t.schemaVersion=="string"?t.schemaVersion:"boid-benchmark-export-v1",generatedAtIso:typeof t.generatedAtIso=="string"?t.generatedAtIso:new Date().toISOString(),filenameBase:typeof t.filenameBase=="string"?t.filenameBase:"imported_benchmark",settings:{boidCount:r.boidCount,separationWeight:r.separationWeight,alignmentWeight:r.alignmentWeight,cohesionWeight:r.cohesionWeight,maxSpeed:r.maxSpeed,updateFrequency:typeof r.updateFrequency=="number"?r.updateFrequency:void 0,projectName:typeof r.projectName=="string"?r.projectName:void 0,groupName:typeof r.groupName=="string"?r.groupName:void 0,version:typeof r.version=="string"?r.version:void 0},hardware:{cpu:s.cpu,gpu:s.gpu,os:s.os},metrics:{avgRenderTime:a.avgRenderTime,avgSimTime:a.avgSimTime},stats:o,frameTimes:n,rollingAverage:{windowSize:60,frameTimes:this.computeRollingAverage(n,60)}}}buildBenchmarkExportPayload(e,t,n,r,s){return{schemaVersion:"boid-benchmark-export-v1",generatedAtIso:r.toISOString(),filenameBase:s,settings:{...t.settings},hardware:{...t.hardware},metrics:{...t.metrics},stats:{...n},frameTimes:[...e],rollingAverage:{windowSize:60,frameTimes:this.computeRollingAverage(e,60)}}}computeRollingAverage(e,t){const n=[];if(t<=0)return n;let r=0;for(let s=0;s<e.length;s++){r+=e[s],s>=t&&(r-=e[s-t]);const a=Math.min(s+1,t);n.push(r/a)}return n}computeStats(e){const t=e.reduce((l,u)=>l+u,0)/e.length,n=1e3/t,r=[...e].sort((l,u)=>u-l),s=Math.max(1,Math.floor(r.length*.01)),a=r.slice(0,s),o=a.reduce((l,u)=>l+u,0)/a.length,c=1e3/o;return{avgFrameTime:t,avgFps:n,onePercentLowFrameTime:o,onePercentLowFps:c}}renderReportCanvas(e,t,n,r){const s=document.createElement("canvas");s.width=2800,s.height=1800;const a=s.getContext("2d");if(!a)throw new Error("Failed to create 2D canvas context.");const o="#FFFFFF",c="#D9D9D9",l="#111111",u="#555555",d="#2F80ED",h="#1A9E55",f="#D23B3B";a.fillStyle=o,a.fillRect(0,0,s.width,s.height),a.strokeStyle=c,a.lineWidth=4,a.strokeRect(8,8,s.width-16,s.height-16);const _=80,y=190,g=340,m=_+y,w=s.height-_-g-m,A=280,U=1500,L=360,N=460,S=30,T=_,G=T+A+S,D=G+U+S,O=D+L+S;return this.drawHeader(a,_,_,s.width-_*2,y,r,t.settings.version??"v1.0.0",l,u),this.drawYAxisGuideBox(a,T,m,A,w,l,u,c),this.drawGraph(a,e,G,m,U,w,l,u,c,d,f),this.drawStatsSidebar(a,D,m,L,w,n,l,u,c,h,f),this.drawNarrativeBox(a,O,m,N,w,l,u,c),this.drawFooter(a,_,s.height-_-g,s.width-_*2,g,t,l,u,c),s}drawHeader(e,t,n,r,s,a,o,c,l){e.fillStyle=c,e.font="700 58px Arial",e.fillText("PERFORMANCE BENCHMARK REPORT",t,n+68),e.fillStyle=l,e.font="500 28px Arial",e.fillText(`Date: ${a.toISOString().slice(0,10)}`,t,n+120),e.fillText(`Version: ${o}`,t+460,n+120),e.fillText("Target: 60Hz (16.67ms)",t+880,n+120),e.strokeStyle="#D9D9D9",e.lineWidth=2,e.beginPath(),e.moveTo(t,n+s-5),e.lineTo(t+r,n+s-5),e.stroke()}drawGraph(e,t,n,r,s,a,o,c,l,u,d){e.strokeStyle=l,e.lineWidth=2,e.strokeRect(n,r,s,a);const h=110,f=20,_=30,y=95,g=n+h,m=r+_,b=s-h-f,w=a-_-y,A=Math.max(22,...t)*1.05,U=[0,8,16.67,24,33,40].filter(O=>O<=A+3);e.strokeStyle="#E9E9E9",e.lineWidth=1;for(const O of U){const V=m+w-O/A*w;e.beginPath(),e.moveTo(g,V),e.lineTo(g+b,V),e.stroke(),e.fillStyle=c,e.font="500 20px Arial",e.fillText(`${O.toFixed(O===16.67?2:0)}ms`,n+12,V+6)}const L=m+w-16.67/A*w;e.strokeStyle=d,e.lineWidth=3,e.setLineDash([12,8]),e.beginPath(),e.moveTo(g,L),e.lineTo(g+b,L),e.stroke(),e.setLineDash([]),e.fillStyle=d,e.font="600 22px Arial",e.fillText("16.67ms (60 FPS)",g+10,L-10),e.strokeStyle=u,e.lineWidth=2.5,e.globalAlpha=.55,e.beginPath();for(let O=0;O<t.length;O++){const V=t.length<=1?0:O/(t.length-1),K=g+V*b,Y=m+w-t[O]/A*w;O===0?e.moveTo(K,Y):e.lineTo(K,Y)}e.stroke(),e.globalAlpha=1;const N=60,S="#F0A500";e.strokeStyle=S,e.lineWidth=3,e.beginPath();let T=!1;for(let O=0;O<t.length;O++){const V=Math.max(0,O-N+1);let K=0;for(let oe=V;oe<=O;oe++)K+=t[oe];const Y=K/(O-V+1),Z=t.length<=1?0:O/(t.length-1),X=g+Z*b,fe=m+w-Y/A*w;T?e.lineTo(X,fe):(e.moveTo(X,fe),T=!0)}e.stroke();const G=g+10,D=m+w-14;e.font="600 20px Arial",e.strokeStyle=u,e.lineWidth=2.5,e.globalAlpha=.55,e.beginPath(),e.moveTo(G,D),e.lineTo(G+36,D),e.stroke(),e.globalAlpha=1,e.fillStyle=u,e.fillText("Frame Time",G+44,D+6),e.strokeStyle=S,e.lineWidth=3,e.beginPath(),e.moveTo(G+220,D),e.lineTo(G+256,D),e.stroke(),e.fillStyle=S,e.fillText("60-Frame Avg",G+264,D+6),e.strokeStyle=d,e.lineWidth=2,e.setLineDash([8,5]),e.beginPath(),e.moveTo(G+460,D),e.lineTo(G+496,D),e.stroke(),e.setLineDash([]),e.fillStyle=d,e.fillText("60 FPS Target",G+504,D+6),e.fillStyle=o,e.font="600 26px Arial",e.fillText("Frame Time (ms)",n+14,r+24),e.fillText("X-Axis: Frame Count / Time",g+b-340,r+a-54),e.save(),e.translate(n+32,r+a/2+120),e.rotate(-Math.PI/2),e.fillText("Frame Time (ms)",0,0),e.restore()}drawYAxisGuideBox(e,t,n,r,s,a,o,c){e.strokeStyle=c,e.lineWidth=2,e.strokeRect(t,n,r,s),e.fillStyle=a,e.font="700 28px Arial",e.fillText("Y-Axis Guide",t+22,n+42);const l=["< 16.67ms: Meets 60 FPS target","16.67-25ms: Minor drops","> 25ms: Noticeable hitching","Spikes: Transient frame stalls","Flat line: Stable pacing"];e.fillStyle=o,e.font="500 21px Arial";let u=n+88;for(const d of l){const h=this.drawWrappedText(e,d,t+22,u,r-40,30);u+=h*30+14}}drawNarrativeBox(e,t,n,r,s,a,o,c){e.strokeStyle=c,e.lineWidth=2,e.strokeRect(t,n,r,s),e.fillStyle=a,e.font="700 28px Arial",e.fillText("Narrative",t+22,n+42);const l=["This chart visualizes per-frame render cadence across a fixed benchmark window.","Short vertical excursions indicate transient frame-time spikes; repeated peaks usually align with compute pressure or memory sync.","A compressed line near or under 16.67ms indicates stable 60 FPS behavior.","Wider variance indicates unstable pacing and inconsistent simulation throughput."];e.fillStyle=o,e.font="500 21px Arial";let u=n+85;for(const d of l){const h=this.drawWrappedText(e,d,t+22,u,r-44,31);u+=h*31+22}}drawStatsSidebar(e,t,n,r,s,a,o,c,l,u,d){e.strokeStyle=l,e.lineWidth=2,e.strokeRect(t,n,r,s),e.fillStyle=o,e.font="700 28px Arial",e.fillText("Statistics",t+20,n+42);const h=a.avgFrameTime<=16.67?u:d,f=a.onePercentLowFrameTime<=16.67?u:d,_=n+70,y=22,g=170;this.drawMetricCard(e,t+18,_,r-36,g,"Avg FPS",a.avgFps.toFixed(1),h,c,l),this.drawMetricCard(e,t+18,_+g+y,r-36,g,"Avg Frame Time",`${a.avgFrameTime.toFixed(2)} ms`,h,c,l),this.drawMetricCard(e,t+18,_+(g+y)*2,r-36,g,"1% Lows",`${a.onePercentLowFps.toFixed(1)} FPS`,f,c,l),e.fillStyle=c,e.font="500 18px Arial",e.fillText(`1% low frame-time: ${a.onePercentLowFrameTime.toFixed(2)} ms`,t+24,n+s-28)}drawMetricCard(e,t,n,r,s,a,o,c,l,u){e.strokeStyle=u,e.lineWidth=2,e.strokeRect(t,n,r,s),e.fillStyle=l,e.font="600 20px Arial",e.fillText(a,t+16,n+34),e.fillStyle=c,e.font="700 44px Arial",e.fillText(o,t+16,n+108)}drawFooter(e,t,n,r,s,a,o,c,l){e.strokeStyle=l,e.lineWidth=2,e.strokeRect(t,n,r,s);const u=r/3;e.beginPath(),e.moveTo(t+u,n),e.lineTo(t+u,n+s),e.moveTo(t+u*2,n),e.lineTo(t+u*2,n+s),e.stroke(),this.drawFooterColumn(e,t+20,n+36,u-40,"Project Details",[`Project Name: ${a.settings.projectName??"Boid Boys"}`,`Group Name: ${a.settings.groupName??"Boid Boys"}`,`Boid Count: ${a.settings.boidCount}`],o,c),this.drawFooterColumn(e,t+u+20,n+36,u-40,"Parameter Settings",[`Separation Weight: ${a.settings.separationWeight.toFixed(2)}`,`Alignment Weight: ${a.settings.alignmentWeight.toFixed(2)}`,`Cohesion Weight: ${a.settings.cohesionWeight.toFixed(2)}`,`Max Speed: ${a.settings.maxSpeed.toFixed(2)}`,`Update Freq: ${a.settings.updateFrequency??"N/A"}`,`Avg Sim: ${a.metrics.avgSimTime.toFixed(2)} ms`,`Avg Render: ${a.metrics.avgRenderTime.toFixed(2)} ms`],o,c),this.drawFooterColumn(e,t+u*2+20,n+36,u-40,"Hardware & Software",["Platform: Chrome / WebGPU"],o,c)}drawFooterColumn(e,t,n,r,s,a,o,c){e.fillStyle=o,e.font="700 28px Arial",e.fillText(s,t,n),e.fillStyle=c,e.font="500 22px Arial";let l=n+42;for(const u of a){const d=this.drawWrappedText(e,u,t,l,r,30);l+=d*30+8}}drawWrappedText(e,t,n,r,s,a){const o=t.split(" ");let c="",l=0;for(let u=0;u<o.length;u++){const d=c?`${c} ${o[u]}`:o[u];e.measureText(d).width>s&&c?(e.fillText(c,n,r),l+=1,c=o[u],r+=a):c=d}return c&&(e.fillText(c,n,r),l+=1),l}formatDateStamp(e){const t=e.getFullYear(),n=String(e.getMonth()+1).padStart(2,"0"),r=String(e.getDate()).padStart(2,"0");return`${t}${n}${r}`}formatBoidLabel(e){return e>=1e3?`${Math.round(e/1e3)}K`:`${e}`}sanitizeFilePart(e){return e.replace(/[/\\?%*:|"<>]/g,"").trim().replace(/\s+/g,"_")}loadRememberedInput(e){try{return localStorage.getItem(e)||""}catch{return""}}saveRememberedInput(e,t){try{localStorage.setItem(e,t)}catch{}}loadRememberedBoolean(e,t){try{const n=localStorage.getItem(e);return n===null?t:n==="true"}catch{return t}}saveRememberedBoolean(e,t){try{localStorage.setItem(e,t?"true":"false")}catch{}}promptBenchmarkNameWithPreview(e,t,n,r){return new Promise(s=>{const a=document.createElement("div");a.style.position="fixed",a.style.inset="0",a.style.background="rgba(0,0,0,0.35)",a.style.display="flex",a.style.alignItems="center",a.style.justifyContent="center",a.style.zIndex="10000";const o=document.createElement("div");o.style.width="520px",o.style.background="#FFFFFF",o.style.border="1px solid #DADADA",o.style.borderRadius="12px",o.style.boxShadow="0 20px 60px rgba(0,0,0,0.2)",o.style.padding="18px",o.style.fontFamily="Arial, sans-serif";const c=document.createElement("h3");c.textContent="Benchmark Export",c.style.margin="0 0 12px 0",c.style.fontSize="20px";const l=document.createElement("input");l.type="text",l.placeholder="Enter label (e.g. RTX4070_TestA)",l.value=this.loadRememberedInput(Xt.STORAGE_KEY_BENCHMARK_NAME),l.style.width="100%",l.style.padding="10px",l.style.border="1px solid #CCCCCC",l.style.borderRadius="8px",l.style.fontSize="16px";const u=document.createElement("div");u.textContent="Illegal filename characters are removed automatically.",u.style.marginTop="8px",u.style.color="#666",u.style.fontSize="13px";const d=document.createElement("label");d.textContent="CPU Type",d.style.display="block",d.style.marginTop="10px",d.style.fontSize="13px",d.style.color="#444";const h=document.createElement("input");h.type="text",h.placeholder="13th Gen Intel(R) Core(TM) i9-13900H",h.value=n||this.loadRememberedInput(Xt.STORAGE_KEY_CPU_TYPE),h.style.width="100%",h.style.padding="10px",h.style.border="1px solid #CCCCCC",h.style.borderRadius="8px",h.style.fontSize="14px";const f=document.createElement("label");f.textContent="GPU Type",f.style.display="block",f.style.marginTop="10px",f.style.fontSize="13px",f.style.color="#444";const _=document.createElement("input");_.type="text",_.placeholder="NVIDIA GeForce RTX 4060 Laptop GPU",_.value=r||this.loadRememberedInput(Xt.STORAGE_KEY_GPU_TYPE),_.style.width="100%",_.style.padding="10px",_.style.border="1px solid #CCCCCC",_.style.borderRadius="8px",_.style.fontSize="14px";const y=document.createElement("div");y.style.marginTop="12px",y.style.padding="10px",y.style.borderRadius="8px",y.style.background="#F5F7FA",y.style.fontSize="14px",y.style.color="#222";const g=document.createElement("label");g.style.marginTop="10px",g.style.display="flex",g.style.alignItems="center",g.style.gap="8px",g.style.fontSize="13px",g.style.color="#444";const m=document.createElement("input");m.type="checkbox",m.checked=this.loadRememberedBoolean(Xt.STORAGE_KEY_EXPORT_JSON,!0);const b=document.createElement("span");b.textContent="Also export JSON with intermediate benchmark data",g.appendChild(m),g.appendChild(b);const w=document.createElement("div");w.style.display="flex",w.style.justifyContent="flex-end",w.style.gap="10px",w.style.marginTop="14px";const A=document.createElement("button");A.textContent="Export LaTeX",A.style.padding="8px 12px",A.style.border="none",A.style.background="#2F80ED",A.style.color="#FFF",A.style.borderRadius="8px",A.style.cursor="pointer";const U=document.createElement("button");U.textContent="Copy LaTeX",U.style.padding="8px 12px",U.style.border="none",U.style.background="#0AA174",U.style.color="#FFF",U.style.borderRadius="8px",U.style.cursor="pointer";const L=()=>{const O=this.sanitizeFilePart(l.value||"Untitled")||"Untitled";y.textContent=`Filename: ${e}_${t}_Boids_${O}.tex`},N=()=>{const G=this.sanitizeFilePart(l.value||"Untitled")||"Untitled",D=h.value.trim(),O=_.value.trim();if(!D||!O)return window.alert("Please fill in both CPU Type and GPU Type."),null;const V=m.checked;return this.saveRememberedInput(Xt.STORAGE_KEY_BENCHMARK_NAME,G),this.saveRememberedInput(Xt.STORAGE_KEY_CPU_TYPE,D),this.saveRememberedInput(Xt.STORAGE_KEY_GPU_TYPE,O),this.saveRememberedBoolean(Xt.STORAGE_KEY_EXPORT_JSON,V),{benchmarkName:G,cpuType:D,gpuType:O,exportJson:V}},S=G=>{window.removeEventListener("keydown",T),a.remove(),s(G)},T=G=>{G.key==="Escape"&&S(null)};l.addEventListener("input",L),l.addEventListener("keydown",G=>{if(G.key==="Enter"){G.preventDefault(),this.setOutputFormats({exportFormat:"latex"});const D=N();if(!D)return;S({...D,action:"export"})}}),A.addEventListener("click",()=>{this.setOutputFormats({exportFormat:"latex"});const G=N();G&&S({...G,action:"export"})}),U.addEventListener("click",()=>{this.setOutputFormats({exportFormat:"latex"});const G=N();G&&S({...G,action:"copy"})}),window.addEventListener("keydown",T),L(),o.appendChild(c),o.appendChild(l),o.appendChild(u),o.appendChild(d),o.appendChild(h),o.appendChild(f),o.appendChild(_),o.appendChild(g),o.appendChild(y),w.appendChild(U),w.appendChild(A),o.appendChild(w),a.appendChild(o),document.body.appendChild(a),l.focus()})}canvasToPngBlob(e){return new Promise((t,n)=>{e.toBlob(r=>{if(!r){n(new Error("Failed to create PNG blob."));return}t(r)},"image/png")})}downloadBlob(e,t){const n=URL.createObjectURL(e),r=document.createElement("a");r.href=n,r.download=t,document.body.appendChild(r),r.click(),document.body.removeChild(r),URL.revokeObjectURL(n)}downloadTextAsBlob(e,t,n){const r=new Blob([e],{type:n});return this.downloadBlob(r,t),r}downloadJson(e,t){const n=JSON.stringify(e,null,2);this.downloadTextAsBlob(n,t,"application/json")}};Xt.STORAGE_KEY_BENCHMARK_NAME="boidBenchmark.benchmarkName",Xt.STORAGE_KEY_CPU_TYPE="boidBenchmark.cpuType",Xt.STORAGE_KEY_GPU_TYPE="boidBenchmark.gpuType",Xt.STORAGE_KEY_EXPORT_JSON="boidBenchmark.exportJson";let to=Xt,Ye,wt,On,En,wc=!0,Gt=1e5,Jn=5e-6,sa=0,lr=[],kl=0,cr=[],hr=[];const zl=new to,no=new $e,aa=new hu;function Yg(i){no.x=i.clientX/window.innerWidth*2-1,no.y=-(i.clientY/window.innerHeight)*2+1}window.addEventListener("mousemove",Yg);async function qg(){if(Ye=new ug,!await Ye.init(Gt,Jn))return;document.getElementById("info-app").innerText="WebGPU Running",wt=new Vg("canvas-container"),await wt.init(),wt.updateVisualBounds(Ye.simulationSize),wt.createInstancedMesh(Gt),On=new Wg(()=>{En.collapseSettingsPanelForBenchmark(),Ye.resetParamsToDefaults(Gt,Jn),Ye.syncParams(),Zg()},()=>{$g(),En.restoreSettingsPanelAfterBenchmark()},zl);const e={getBoidCount:()=>Gt,getBoidDensity:()=>Jn,onRecreateBoids:(t,n)=>{Gt=t,Jn=n,Ye.recreateBoids(t,n),wt.updateVisualBounds(Ye.simulationSize),wt.createInstancedMesh(Gt),En.updateInfo(Gt,Ye.numCells)},onUpdateUniforms:t=>{const n=Ye.numCells;Ye.paramsArray[Be.SEPARATION_DIST]=t.separation,Ye.paramsArray[Be.ALIGN_DIST]=t.align,Ye.paramsArray[Be.COHESION_DIST]=t.cohesion,Ye.paramsArray[Be.MAX_SPEED]=t.max_speed,Ye.paramsArray[Be.MAX_FORCE]=t.max_force,Ye.paramsArray[Be.SEPARATION_WEIGHT]=t.sep_weight,Ye.paramsArray[Be.ALIGNMENT_WEIGHT]=t.align_weight,Ye.paramsArray[Be.COHESION_WEIGHT]=t.coh_weight,Ye.paramsArray[Be.MARGIN]=t.margin,Ye.paramsArray[Be.TURN_FACTOR]=t.turn_factor,typeof t.vision_angle<"u"&&(Ye.paramsArray[Be.VISION_ANGLE]=t.vision_angle*Math.PI/180),Ye.syncParams(),Ye.numCells!==n&&(Ye.initSpatialHashBuffers(),Ye.createBindGroups())},onSimulationToggle:t=>{wc=t},onResetSimulation:()=>{Gt=1e5,Jn=5e-6,Ye.recreateBoids(Gt,Jn),wt.updateVisualBounds(Ye.simulationSize),wt.createInstancedMesh(Gt),En.populateInputs({boidCount:Gt,boidDensity:Jn,params:Ye.paramsArray})},onBenchmarkStart:()=>On.start(),onImportReport:async t=>await zl.openBenchmarkPreviewFromTexFiles(t)};En=new Xg(e),En.init({boidCount:Gt,boidDensity:Jn,params:Ye.paramsArray}),En.updateInfo(Gt,Ye.numCells),Cc()}function Zg(){if(!wt.camera||!wt.controls)return;const i=Ye.simulationSize.x/2,e=Ye.simulationSize.y/2,t=Ye.simulationSize.z/2,n=Ye.simulationSize.x*1.5;wt.camera.position.set(i+n,e+n,t+n),wt.controls.target.set(i,e,t),wt.controls.update(),wt.controls.enabled=!1}function $g(){wt.controls&&(wt.controls.enabled=!0)}async function Cc(){requestAnimationFrame(Cc);const i=performance.now();if(On.recordFrame(i),En.updateBenchmarkHUD(On.getStatus(i)),sa){const e=i-sa;if(lr.push(e),lr.length>og&&lr.shift(),i-kl>=lg){const n=1e3/(lr.reduce((a,o)=>a+o,0)/lr.length);let r=null,s=null;cr.length>0&&(r=cr.reduce((a,o)=>a+o,0)/cr.length),hr.length>0&&(s=hr.reduce((a,o)=>a+o,0)/hr.length),En.updateFPS(n,r,s),kl=i,cr=[],hr=[]}}if(sa=i,En.updateInfo(Gt,Ye.numCells),wc){aa.setFromCamera(no,wt.camera);const e=aa.ray.origin,t=aa.ray.direction;Ye.paramsArray[Be.MOUSE_RAY_ORIGIN]=e.x,Ye.paramsArray[Be.MOUSE_RAY_ORIGIN+1]=e.y,Ye.paramsArray[Be.MOUSE_RAY_ORIGIN+2]=e.z,Ye.paramsArray[Be.RAY_DIRECTION]=t.x,Ye.paramsArray[Be.RAY_DIRECTION+1]=t.y,Ye.paramsArray[Be.RAY_DIRECTION+2]=t.z,Ye.paramsArray[Be.FLEE_RADIUS]=Ye.simulationSize.x*.12;const n=await Ye.step(wt);n&&(cr.push(n.simDelta),On.recordSimulationSample(n.simDelta),hr.push(n.renderDelta),On.recordRenderSample(n.renderDelta))}if(On.state===2&&performance.now()>On.recordEndsAt){const e={boidCount:Gt,separationWeight:Ye.paramsArray[Be.SEPARATION_WEIGHT],alignmentWeight:Ye.paramsArray[Be.ALIGNMENT_WEIGHT],cohesionWeight:Ye.paramsArray[Be.COHESION_WEIGHT],maxSpeed:Ye.paramsArray[Be.MAX_SPEED],updateFrequency:WORKGROUP_SIZE,projectName:"Boid Boys",groupName:"Boid Boys",version:"v1.0.0"};On.completeBenchmark(e)}wt.render(i)}qg();
