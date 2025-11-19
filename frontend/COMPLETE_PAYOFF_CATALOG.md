# Complete Payoff Catalog - All 360 Payoffs

This document lists ALL 360 unique payoffs from the optimal_stopping Python codebase.

## Total Count: 360 Payoffs

- **30 Base Payoffs** (no barriers)
- **330 Barrier Variants** (30 base × 11 barrier types)

---

## PART 1: BASE PAYOFFS (30 total)

### SINGLE ASSET (12 payoffs)

#### 1. Simple (2)
1. **Call** - European Call: max(0, S - K)
2. **Put** - European Put: max(0, K - S)

#### 2. Lookback (4)
3. **LookbackFixedCall** - max(0, max_over_time(S) - K)
4. **LookbackFixedPut** - max(0, K - min_over_time(S))
5. **LookbackFloatCall** - max(0, S(T) - min_over_time(S))
6. **LookbackFloatPut** - max(0, max_over_time(S) - S(T))

#### 3. Asian (4)
7. **AsianFixedStrikeCall_Single** - max(0, avg_over_time(S) - K)
8. **AsianFixedStrikePut_Single** - max(0, K - avg_over_time(S))
9. **AsianFloatingStrikeCall_Single** - max(0, S(T) - avg_over_time(S))
10. **AsianFloatingStrikePut_Single** - max(0, avg_over_time(S) - S(T))

#### 4. Range (2)
11. **RangeCall_Single** - max(0, [max_over_time(S) - min_over_time(S)] - K)
12. **RangePut_Single** - max(0, K - [max_over_time(S) - min_over_time(S)])

---

### BASKET (18 payoffs)

#### 5. Simple (6)
13. **BasketCall** - max(0, mean(S) - K)
14. **BasketPut** - max(0, K - mean(S))
15. **GeometricCall** - max(0, geom_mean(S) - K)
16. **GeometricPut** - max(0, K - geom_mean(S))
17. **MaxCall** - max(0, max(S_i) - K)
18. **MinPut** - max(0, K - min(S_i))

#### 6. Asian (4)
19. **AsianFixedStrikeCall** - max(0, avg_over_time(mean(S)) - K)
20. **AsianFixedStrikePut** - max(0, K - avg_over_time(mean(S)))
21. **AsianFloatingStrikeCall** - max(0, mean(S_T) - avg_over_time(mean(S)))
22. **AsianFloatingStrikePut** - max(0, avg_over_time(mean(S)) - mean(S_T))

#### 7. Dispersion (4)
23. **MaxDispersionCall** - max(0, [max_i,t(S_i,t) - min_i,t(S_i,t)] - K)
24. **MaxDispersionPut** - max(0, K - [max_i,t(S_i,t) - min_i,t(S_i,t)])
25. **DispersionCall** - max(0, σ(S) - K)
26. **DispersionPut** - max(0, K - σ(S))

#### 8. Rank (4)
27. **BestOfKCall** - max(0, mean(top_k_prices) - K)
28. **WorstOfKPut** - max(0, K - mean(bottom_k_prices))
29. **RankWeightedBasketCall** - max(0, sum(w_i * S_(i)) - K)
30. **RankWeightedBasketPut** - max(0, K - sum(w_i * S_(i)))

---

## PART 2: BARRIER VARIANTS (330 total)

Each of the 30 base payoffs has 11 barrier variants.

### Barrier Types (11)

1. **UO** - Up-and-Out
2. **DO** - Down-and-Out
3. **UI** - Up-and-In
4. **DI** - Down-and-In
5. **UODO** - Double Knock-Out
6. **UIDI** - Double Knock-In
7. **UIDO** - Up-In-Down-Out
8. **UODI** - Up-Out-Down-In
9. **PTB** - Partial Time Barrier
10. **StepB** - Step Barrier
11. **DStepB** - Double Step Barrier

---

### Complete List of 330 Barrier Variants

#### SINGLE ASSET - Call (11 variants)
31. UO_Call
32. DO_Call
33. UI_Call
34. DI_Call
35. UODO_Call
36. UIDI_Call
37. UIDO_Call
38. UODI_Call
39. PTB_Call
40. StepB_Call
41. DStepB_Call

#### SINGLE ASSET - Put (11 variants)
42. UO_Put
43. DO_Put
44. UI_Put
45. DI_Put
46. UODO_Put
47. UIDI_Put
48. UIDO_Put
49. UODI_Put
50. PTB_Put
51. StepB_Put
52. DStepB_Put

#### SINGLE ASSET - LookbackFixedCall (11 variants)
53. UO_LookbackFixedCall
54. DO_LookbackFixedCall
55. UI_LookbackFixedCall
56. DI_LookbackFixedCall
57. UODO_LookbackFixedCall
58. UIDI_LookbackFixedCall
59. UIDO_LookbackFixedCall
60. UODI_LookbackFixedCall
61. PTB_LookbackFixedCall
62. StepB_LookbackFixedCall
63. DStepB_LookbackFixedCall

#### SINGLE ASSET - LookbackFixedPut (11 variants)
64. UO_LookbackFixedPut
65. DO_LookbackFixedPut
66. UI_LookbackFixedPut
67. DI_LookbackFixedPut
68. UODO_LookbackFixedPut
69. UIDI_LookbackFixedPut
70. UIDO_LookbackFixedPut
71. UODI_LookbackFixedPut
72. PTB_LookbackFixedPut
73. StepB_LookbackFixedPut
74. DStepB_LookbackFixedPut

#### SINGLE ASSET - LookbackFloatCall (11 variants)
75. UO_LookbackFloatCall
76. DO_LookbackFloatCall
77. UI_LookbackFloatCall
78. DI_LookbackFloatCall
79. UODO_LookbackFloatCall
80. UIDI_LookbackFloatCall
81. UIDO_LookbackFloatCall
82. UODI_LookbackFloatCall
83. PTB_LookbackFloatCall
84. StepB_LookbackFloatCall
85. DStepB_LookbackFloatCall

#### SINGLE ASSET - LookbackFloatPut (11 variants)
86. UO_LookbackFloatPut
87. DO_LookbackFloatPut
88. UI_LookbackFloatPut
89. DI_LookbackFloatPut
90. UODO_LookbackFloatPut
91. UIDI_LookbackFloatPut
92. UIDO_LookbackFloatPut
93. UODI_LookbackFloatPut
94. PTB_LookbackFloatPut
95. StepB_LookbackFloatPut
96. DStepB_LookbackFloatPut

#### SINGLE ASSET - AsianFixedStrikeCall_Single (11 variants)
97. UO_AsianFixedStrikeCall_Single
98. DO_AsianFixedStrikeCall_Single
99. UI_AsianFixedStrikeCall_Single
100. DI_AsianFixedStrikeCall_Single
101. UODO_AsianFixedStrikeCall_Single
102. UIDI_AsianFixedStrikeCall_Single
103. UIDO_AsianFixedStrikeCall_Single
104. UODI_AsianFixedStrikeCall_Single
105. PTB_AsianFixedStrikeCall_Single
106. StepB_AsianFixedStrikeCall_Single
107. DStepB_AsianFixedStrikeCall_Single

#### SINGLE ASSET - AsianFixedStrikePut_Single (11 variants)
108. UO_AsianFixedStrikePut_Single
109. DO_AsianFixedStrikePut_Single
110. UI_AsianFixedStrikePut_Single
111. DI_AsianFixedStrikePut_Single
112. UODO_AsianFixedStrikePut_Single
113. UIDI_AsianFixedStrikePut_Single
114. UIDO_AsianFixedStrikePut_Single
115. UODI_AsianFixedStrikePut_Single
116. PTB_AsianFixedStrikePut_Single
117. StepB_AsianFixedStrikePut_Single
118. DStepB_AsianFixedStrikePut_Single

#### SINGLE ASSET - AsianFloatingStrikeCall_Single (11 variants)
119. UO_AsianFloatingStrikeCall_Single
120. DO_AsianFloatingStrikeCall_Single
121. UI_AsianFloatingStrikeCall_Single
122. DI_AsianFloatingStrikeCall_Single
123. UODO_AsianFloatingStrikeCall_Single
124. UIDI_AsianFloatingStrikeCall_Single
125. UIDO_AsianFloatingStrikeCall_Single
126. UODI_AsianFloatingStrikeCall_Single
127. PTB_AsianFloatingStrikeCall_Single
128. StepB_AsianFloatingStrikeCall_Single
129. DStepB_AsianFloatingStrikeCall_Single

#### SINGLE ASSET - AsianFloatingStrikePut_Single (11 variants)
130. UO_AsianFloatingStrikePut_Single
131. DO_AsianFloatingStrikePut_Single
132. UI_AsianFloatingStrikePut_Single
133. DI_AsianFloatingStrikePut_Single
134. UODO_AsianFloatingStrikePut_Single
135. UIDI_AsianFloatingStrikePut_Single
136. UIDO_AsianFloatingStrikePut_Single
137. UODI_AsianFloatingStrikePut_Single
138. PTB_AsianFloatingStrikePut_Single
139. StepB_AsianFloatingStrikePut_Single
140. DStepB_AsianFloatingStrikePut_Single

#### SINGLE ASSET - RangeCall_Single (11 variants)
141. UO_RangeCall_Single
142. DO_RangeCall_Single
143. UI_RangeCall_Single
144. DI_RangeCall_Single
145. UODO_RangeCall_Single
146. UIDI_RangeCall_Single
147. UIDO_RangeCall_Single
148. UODI_RangeCall_Single
149. PTB_RangeCall_Single
150. StepB_RangeCall_Single
151. DStepB_RangeCall_Single

#### SINGLE ASSET - RangePut_Single (11 variants)
152. UO_RangePut_Single
153. DO_RangePut_Single
154. UI_RangePut_Single
155. DI_RangePut_Single
156. UODO_RangePut_Single
157. UIDI_RangePut_Single
158. UIDO_RangePut_Single
159. UODI_RangePut_Single
160. PTB_RangePut_Single
161. StepB_RangePut_Single
162. DStepB_RangePut_Single

#### BASKET - BasketCall (11 variants)
163. UO_BasketCall
164. DO_BasketCall
165. UI_BasketCall
166. DI_BasketCall
167. UODO_BasketCall
168. UIDI_BasketCall
169. UIDO_BasketCall
170. UODI_BasketCall
171. PTB_BasketCall
172. StepB_BasketCall
173. DStepB_BasketCall

#### BASKET - BasketPut (11 variants)
174. UO_BasketPut
175. DO_BasketPut
176. UI_BasketPut
177. DI_BasketPut
178. UODO_BasketPut
179. UIDI_BasketPut
180. UIDO_BasketPut
181. UODI_BasketPut
182. PTB_BasketPut
183. StepB_BasketPut
184. DStepB_BasketPut

#### BASKET - GeometricCall (11 variants)
185. UO_GeometricCall
186. DO_GeometricCall
187. UI_GeometricCall
188. DI_GeometricCall
189. UODO_GeometricCall
190. UIDI_GeometricCall
191. UIDO_GeometricCall
192. UODI_GeometricCall
193. PTB_GeometricCall
194. StepB_GeometricCall
195. DStepB_GeometricCall

#### BASKET - GeometricPut (11 variants)
196. UO_GeometricPut
197. DO_GeometricPut
198. UI_GeometricPut
199. DI_GeometricPut
200. UODO_GeometricPut
201. UIDI_GeometricPut
202. UIDO_GeometricPut
203. UODI_GeometricPut
204. PTB_GeometricPut
205. StepB_GeometricPut
206. DStepB_GeometricPut

#### BASKET - MaxCall (11 variants)
207. UO_MaxCall
208. DO_MaxCall
209. UI_MaxCall
210. DI_MaxCall
211. UODO_MaxCall
212. UIDI_MaxCall
213. UIDO_MaxCall
214. UODI_MaxCall
215. PTB_MaxCall
216. StepB_MaxCall
217. DStepB_MaxCall

#### BASKET - MinPut (11 variants)
218. UO_MinPut
219. DO_MinPut
220. UI_MinPut
221. DI_MinPut
222. UODO_MinPut
223. UIDI_MinPut
224. UIDO_MinPut
225. UODI_MinPut
226. PTB_MinPut
227. StepB_MinPut
228. DStepB_MinPut

#### BASKET - AsianFixedStrikeCall (11 variants)
229. UO_AsianFixedStrikeCall
230. DO_AsianFixedStrikeCall
231. UI_AsianFixedStrikeCall
232. DI_AsianFixedStrikeCall
233. UODO_AsianFixedStrikeCall
234. UIDI_AsianFixedStrikeCall
235. UIDO_AsianFixedStrikeCall
236. UODI_AsianFixedStrikeCall
237. PTB_AsianFixedStrikeCall
238. StepB_AsianFixedStrikeCall
239. DStepB_AsianFixedStrikeCall

#### BASKET - AsianFixedStrikePut (11 variants)
240. UO_AsianFixedStrikePut
241. DO_AsianFixedStrikePut
242. UI_AsianFixedStrikePut
243. DI_AsianFixedStrikePut
244. UODO_AsianFixedStrikePut
245. UIDI_AsianFixedStrikePut
246. UIDO_AsianFixedStrikePut
247. UODI_AsianFixedStrikePut
248. PTB_AsianFixedStrikePut
249. StepB_AsianFixedStrikePut
250. DStepB_AsianFixedStrikePut

#### BASKET - AsianFloatingStrikeCall (11 variants)
251. UO_AsianFloatingStrikeCall
252. DO_AsianFloatingStrikeCall
253. UI_AsianFloatingStrikeCall
254. DI_AsianFloatingStrikeCall
255. UODO_AsianFloatingStrikeCall
256. UIDI_AsianFloatingStrikeCall
257. UIDO_AsianFloatingStrikeCall
258. UODI_AsianFloatingStrikeCall
259. PTB_AsianFloatingStrikeCall
260. StepB_AsianFloatingStrikeCall
261. DStepB_AsianFloatingStrikeCall

#### BASKET - AsianFloatingStrikePut (11 variants)
262. UO_AsianFloatingStrikePut
263. DO_AsianFloatingStrikePut
264. UI_AsianFloatingStrikePut
265. DI_AsianFloatingStrikePut
266. UODO_AsianFloatingStrikePut
267. UIDI_AsianFloatingStrikePut
268. UIDO_AsianFloatingStrikePut
269. UODI_AsianFloatingStrikePut
270. PTB_AsianFloatingStrikePut
271. StepB_AsianFloatingStrikePut
272. DStepB_AsianFloatingStrikePut

#### BASKET - MaxDispersionCall (11 variants)
273. UO_MaxDispersionCall
274. DO_MaxDispersionCall
275. UI_MaxDispersionCall
276. DI_MaxDispersionCall
277. UODO_MaxDispersionCall
278. UIDI_MaxDispersionCall
279. UIDO_MaxDispersionCall
280. UODI_MaxDispersionCall
281. PTB_MaxDispersionCall
282. StepB_MaxDispersionCall
283. DStepB_MaxDispersionCall

#### BASKET - MaxDispersionPut (11 variants)
284. UO_MaxDispersionPut
285. DO_MaxDispersionPut
286. UI_MaxDispersionPut
287. DI_MaxDispersionPut
288. UODO_MaxDispersionPut
289. UIDI_MaxDispersionPut
290. UIDO_MaxDispersionPut
291. UODI_MaxDispersionPut
292. PTB_MaxDispersionPut
293. StepB_MaxDispersionPut
294. DStepB_MaxDispersionPut

#### BASKET - DispersionCall (11 variants)
295. UO_DispersionCall
296. DO_DispersionCall
297. UI_DispersionCall
298. DI_DispersionCall
299. UODO_DispersionCall
300. UIDI_DispersionCall
301. UIDO_DispersionCall
302. UODI_DispersionCall
303. PTB_DispersionCall
304. StepB_DispersionCall
305. DStepB_DispersionCall

#### BASKET - DispersionPut (11 variants)
306. UO_DispersionPut
307. DO_DispersionPut
308. UI_DispersionPut
309. DI_DispersionPut
310. UODO_DispersionPut
311. UIDI_DispersionPut
312. UIDO_DispersionPut
313. UODI_DispersionPut
314. PTB_DispersionPut
315. StepB_DispersionPut
316. DStepB_DispersionPut

#### BASKET - BestOfKCall (11 variants)
317. UO_BestOfKCall
318. DO_BestOfKCall
319. UI_BestOfKCall
320. DI_BestOfKCall
321. UODO_BestOfKCall
322. UIDI_BestOfKCall
323. UIDO_BestOfKCall
324. UODI_BestOfKCall
325. PTB_BestOfKCall
326. StepB_BestOfKCall
327. DStepB_BestOfKCall

#### BASKET - WorstOfKPut (11 variants)
328. UO_WorstOfKPut
329. DO_WorstOfKPut
330. UI_WorstOfKPut
331. DI_WorstOfKPut
332. UODO_WorstOfKPut
333. UIDI_WorstOfKPut
334. UIDO_WorstOfKPut
335. UODI_WorstOfKPut
336. PTB_WorstOfKPut
337. StepB_WorstOfKPut
338. DStepB_WorstOfKPut

#### BASKET - RankWeightedBasketCall (11 variants)
339. UO_RankWeightedBasketCall
340. DO_RankWeightedBasketCall
341. UI_RankWeightedBasketCall
342. DI_RankWeightedBasketCall
343. UODO_RankWeightedBasketCall
344. UIDI_RankWeightedBasketCall
345. UIDO_RankWeightedBasketCall
346. UODI_RankWeightedBasketCall
347. PTB_RankWeightedBasketCall
348. StepB_RankWeightedBasketCall
349. DStepB_RankWeightedBasketCall

#### BASKET - RankWeightedBasketPut (11 variants)
350. UO_RankWeightedBasketPut
351. DO_RankWeightedBasketPut
352. UI_RankWeightedBasketPut
353. DI_RankWeightedBasketPut
354. UODO_RankWeightedBasketPut
355. UIDI_RankWeightedBasketPut
356. UIDO_RankWeightedBasketPut
357. UODI_RankWeightedBasketPut
358. PTB_RankWeightedBasketPut
359. StepB_RankWeightedBasketPut
360. DStepB_RankWeightedBasketPut

---

## TOTAL: 360 UNIQUE PAYOFFS

✓ 30 base payoffs
✓ 330 barrier variants (30 × 11)
✓ All implemented in TypeScript
✓ All accessible via PayoffSelector component
