# scibowl-gpt
Finetuning an open weights model for automatically generating Science Bowl questions. 

## Future Todo:
- Improve packet parser
- Make model to put questions into subcategories 
- Create reward model on question quality for RLHF

## Example prompt
**Input:**
```
Category: Math, Type: Tossup, Subcategory: Combinatorics, Difficulty: 3
```

**Output:**
```
Tim flips a fair coin 10 times. Given that the first 3 flips all landed on the same side, what is the expected number of total heads flipped?

ANSWER: 5
```

### Subcategories:
**Math:** Algebra, Geometry, Combinatorics, Number Theory, Calculus, Other
**Physics:** Kinematics, E+M, Thermodynamics, Optics, Quantum, Particle, Waves, Relativity
**Earth and Space:** Hydrology, Tectonics/Volcanism, Meteorology, Rocks & Minerals, Cosmology, Solar System, Stars, Observational Astronomy
**Biology:** Biochemistry, Cell/Molecular Biology, Genetics/Evolution, Plants, Animals, Ecology, Biosystematics, Physiology
**Chemistry:** [insert good list here]
**Energy:** Physics, Earth, Space, Biology, Chemistry, Machine Learning, Stats, Theoretical CS

### Difficulty:
- 0: regionals RR
- 1: regionals RR
- 2: Nats RR, Regionals early DE
- 3: Nats RR, Regionals mid DE
- 4: Nats Early DE, Regionals late DE
- 5: Nats mid DE
- 6: Nats late DE
- 7: Too hard for Science Bowl