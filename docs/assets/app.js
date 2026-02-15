// ============================================================================
// ANES 2024 Ideological Clusters - Main Application
// ============================================================================

(function() {
    'use strict';

    // ========================================================================
    // GLOBAL STATE
    // ========================================================================

    const STATE = {
        metadata: null,
        clusters: null,
        preprocess: null,
        centroids: null,
        distances: null,
        embedding2d: null,
        story: null,
        avatars: null,
        mlPredictions: null,
        welcomeTour: null,
        currentTab: 'explore',
        currentViz: '3d',
        currentStoryStep: 0,
        currentPersona: null,
        quizAnswers: {},
        chatHistory: []
    };

    // Full question text for each dimension (from ANES 2024 survey)
    const QUESTION_TEXT = {
        "V241248": "Abortion: Do you think by law abortion should always be permitted (1), never be permitted (7), or something in between?",
        "V241386": "Immigration: What should happen to unauthorized immigrants who are currently living in the United States? Should they be given felony status and be deported (1), or should they not be penalized at all (5)?",
        "V241239": "Government services and spending: Some people think the government should provide fewer services even in areas such as health and education in order to reduce spending (1). Other people feel it is important for the government to provide many more services even if it means an increase in spending (7).",
        "V241258": "Environment: Some people think it is important to protect the environment even if it costs some jobs or otherwise reduces our standard of living (1). Other people think that protecting the environment is not as important as maintaining jobs and our standard of living (7).",
        "V241245": "Health insurance: Do you favor a government insurance plan (1), a private insurance plan (7), or something in between?",
        "V241242": "Defense spending: Should federal spending on defense be decreased (1), increased (7), or kept about the same (4)?",
        "V241275x": "Welfare spending: Should federal spending on welfare programs be increased (1), decreased (7), or kept about the same (4)?",
        "V241726": "Trust in government: People like me don't have any say about what the government does. Do you agree (1) or disagree (5)?",
        "V241731": "Importance of democracy: How important is it for the United States to be a democracy? Extremely important (1), very important (2), moderately important (3), or not at all important (4)?",
        "V241732": "Vaccine mandates: Do you favor (1) or oppose (2) requiring children to be vaccinated in order to attend public schools?"
    };

    // ========================================================================
    // DATA LOADING
    // ========================================================================

    async function loadAllData() {
        try {
            const [metadata, profiles, preprocess, centroids, distances, embedding, story, avatars, ml, tour, quizFeats, questionMappings] = await Promise.all([
                fetch('data/metadata.json').then(r => r.json()),
                fetch('data/cluster_profiles.json').then(r => r.json()),
                fetch('data/preprocess.json').then(r => r.json()),
                fetch('data/centroids.json').then(r => r.json()),
                fetch('data/cluster_distances.json').then(r => r.json()),
                fetch('data/embedding_2d.json').then(r => r.json()),
                fetch('data/story.json').then(r => r.json()),
                fetch('data/avatars.json').then(r => r.json()),
                fetch('data/ml_holdout_predictions.json').then(r => r.json()),
                fetch('data/welcome_tour.json').then(r => r.json()),
                fetch('data/quiz_features.json').then(r => r.json()),
                fetch('data/question_mappings.json').then(r => r.json())
            ]);

            STATE.metadata = metadata;
            STATE.clusters = profiles.clusters;
            STATE.preprocess = preprocess;
            STATE.centroids = centroids.centroids;
            STATE.distances = distances.distances;
            STATE.embedding2d = embedding.clusters;
            STATE.story = story;
            STATE.avatars = avatars.personas;
            STATE.mlPredictions = ml;
            STATE.welcomeTour = tour;
            STATE.quizFeatures = quizFeats.quiz_features;
            STATE.questionMappings = questionMappings;

            console.log('All data loaded successfully');
            initializeApp();
        } catch (error) {
            console.error('Error loading data:', error);
            alert('Failed to load data. Please check that all JSON files are present in data/ folder.');
        }
    }

    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    function initializeApp() {
        setupTabs();
        setupExploreTab();
        setupQuizTab();
        // setupStoryTab(); // Removed - Story Mode tab no longer exists
        setupPersonaTab();
        setupDistributionListeners();
        setupWelcomeTour();
        trackEvent('page_load', { universe: STATE.metadata.analysis_universe });
    }

    // ========================================================================
    // TAB SWITCHING
    // ========================================================================

    function setupTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');

        // Expose global function for tour
        window.switchToTab = function(tabName) {
            const targetBtn = document.querySelector(`[data-tab="${tabName}"]`);
            if (targetBtn) targetBtn.click();
        };

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;

                // Update buttons
                tabBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Update content
                tabContents.forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(`${targetTab}-content`).classList.add('active');

                STATE.currentTab = targetTab;
                trackEvent('tab_opened', { tab: targetTab });
            });
        });
    }

    // ========================================================================
    // EXPLORE TAB - 3D CUBE & 2D MAP
    // ========================================================================

    function setupExploreTab() {
        const vizToggles = document.querySelectorAll('.viz-toggle');
        vizToggles.forEach(toggle => {
            toggle.addEventListener('click', () => {
                const viz = toggle.dataset.viz;
                vizToggles.forEach(t => t.classList.remove('active'));
                toggle.classList.add('active');

                document.getElementById('viz-3d').classList.toggle('hidden', viz !== '3d');
                document.getElementById('viz-2d').classList.toggle('hidden', viz !== '2d');

                STATE.currentViz = viz;
                trackEvent(viz === '3d' ? 'opened_3d_cube' : 'opened_2d_map');

                if (viz === '3d') {
                    render3DCube();
                } else {
                    render2DMap();
                }
            });
        });

        // Initialize 3D cube
        render3DCube();

        // Populate legend
        renderClusterLegend();
    }

    function render3DCube() {
        const features = STATE.preprocess.features;
        const dimX = document.getElementById('dim-x');
        const dimY = document.getElementById('dim-y');
        const dimZ = document.getElementById('dim-z');

        // Populate selectors (if empty)
        if (dimX.options.length === 0) {
            features.forEach((f, i) => {
                const readableName = getReadableFeatureName(f);
                [dimX, dimY, dimZ].forEach(sel => {
                    const opt = document.createElement('option');
                    opt.value = f;
                    opt.textContent = readableName;
                    sel.appendChild(opt);
                });
            });

            // Set defaults
            if (features.length >= 3) {
                dimX.value = features[0];
                dimY.value = features[1];
                dimZ.value = features[2];
            }

            // Add change listeners
            [dimX, dimY, dimZ].forEach(sel => {
                sel.addEventListener('change', () => {
                    render3DCube();
                    trackEvent('cube_axis_changed');
                });
            });
        }

        const xVar = dimX.value;
        const yVar = dimY.value;
        const zVar = dimZ.value;

        // Extract data
        const x = STATE.clusters.map(c => c[`centroid_${xVar}`]);
        const y = STATE.clusters.map(c => c[`centroid_${yVar}`]);
        const z = STATE.clusters.map(c => c[`centroid_${zVar}`]);

        // Marker sizes (proportional to pop_share)
        const sizes = STATE.clusters.map(c => 8 + (c.pop_share * 100) * 0.5);

        // Colors (party ID) - RED = Republican (high), BLUE = Democrat (low)
        const colors = STATE.clusters.map(c => c.partyid_mean || 4);

        // Hover text
        const hoverTexts = STATE.clusters.map(c => buildHoverText(c));

        const trace = {
            x, y, z,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: sizes,
                color: colors,
                colorscale: 'RdBu',  // Red = Republican (7), Blue = Democrat (1)
                reversescale: false,  // Red for high values (Republican)
                showscale: true,
                colorbar: { title: 'Party ID<br>(1=Dem, 7=Rep)' }
            },
            text: hoverTexts,
            hovertemplate: '%{text}<extra></extra>'
        };

        const layout = {
            title: `3D Cluster Map: ${getReadableFeatureName(xVar)} × ${getReadableFeatureName(yVar)} × ${getReadableFeatureName(zVar)}`,
            scene: {
                xaxis: { title: getReadableFeatureName(xVar) },
                yaxis: { title: getReadableFeatureName(yVar) },
                zaxis: { title: getReadableFeatureName(zVar) },
                aspectmode: 'cube'
            },
            margin: { l: 0, r: 0, t: 40, b: 0 },
            hovermode: 'closest'
        };

        Plotly.newPlot('plot-3d', [trace], layout, { responsive: true });

        // Update dimension questions display
        updateDimensionQuestions(xVar, yVar, zVar);
    }

    function updateDimensionQuestions(xVar, yVar, zVar) {
        const questionDiv = document.getElementById('dimension-questions');
        if (!questionDiv) return;

        // Use question mappings if available, fallback to hardcoded QUESTION_TEXT
        const xText = getQuizQuestionText(xVar) || QUESTION_TEXT[xVar] || 'Question text not available';
        const yText = getQuizQuestionText(yVar) || QUESTION_TEXT[yVar] || 'Question text not available';
        const zText = getQuizQuestionText(zVar) || QUESTION_TEXT[zVar] || 'Question text not available';

        questionDiv.innerHTML = `
            <div class="question-item">
                <strong>X-axis (${getReadableFeatureName(xVar)}):</strong> ${xText}
            </div>
            <div class="question-item">
                <strong>Y-axis (${getReadableFeatureName(yVar)}):</strong> ${yText}
            </div>
            <div class="question-item">
                <strong>Z-axis (${getReadableFeatureName(zVar)}):</strong> ${zText}
            </div>
        `;
    }

    function render2DMap() {
        const x = STATE.clusters.map(c => STATE.embedding2d[c.cluster].x);
        const y = STATE.clusters.map(c => STATE.embedding2d[c.cluster].y);
        const sizes = STATE.clusters.map(c => 10 + (c.pop_share * 100) * 0.6);
        const colors = STATE.clusters.map(c => c.partyid_mean || 4);
        const hoverTexts = STATE.clusters.map(c => buildHoverText(c));

        const trace = {
            x, y,
            mode: 'markers+text',
            type: 'scatter',
            marker: {
                size: sizes,
                color: colors,
                colorscale: 'RdBu',  // Red = Republican (7), Blue = Democrat (1)
                reversescale: false,  // Red for high values (Republican)
                showscale: true,
                colorbar: { title: 'Party ID<br>(1=Dem, 7=Rep)' }
            },
            text: STATE.clusters.map(c => `C${c.cluster}`),
            textposition: 'top center',
            hovertext: hoverTexts,
            hovertemplate: '%{hovertext}<extra></extra>'
        };

        const layout = {
            title: `2D PCA Map (Variance: ${(STATE.embedding2d.explained_variance || [0,0]).map(v => (v*100).toFixed(1)).join('%, ')}%)`,
            xaxis: { title: 'PC1', zeroline: true },
            yaxis: { title: 'PC2', zeroline: true },
            hovermode: 'closest',
            height: 600
        };

        Plotly.newPlot('plot-2d', [trace], layout, { responsive: true });
    }

    function buildHoverText(cluster) {
        const parts = [
            `<b>Cluster ${cluster.cluster}</b>`,
            `Pop Share: ${(cluster.pop_share * 100).toFixed(1)}% (n=${cluster.n_unweighted})`,
            cluster.partyid_mean ? `Party ID: ${cluster.partyid_mean.toFixed(2)}` : '',
            cluster.vote_harris ? `Harris: ${(cluster.vote_harris * 100).toFixed(1)}% | Trump: ${(cluster.vote_trump * 100).toFixed(1)}% | Other: ${(cluster.vote_other * 100).toFixed(1)}%` : '',
            cluster.demographics.gender_Man ? `Gender: ${(cluster.demographics.gender_Man * 100).toFixed(0)}% M / ${(cluster.demographics.gender_Woman * 100).toFixed(0)}% W` : '',
            cluster.region.South ? `Region: NE ${(cluster.region.Northeast * 100).toFixed(0)}% / MW ${(cluster.region.Midwest * 100).toFixed(0)}% / S ${(cluster.region.South * 100).toFixed(0)}% / W ${(cluster.region.West * 100).toFixed(0)}%` : '',
            cluster.religion.mean_attendance ? `Religion: Attendance ${cluster.religion.mean_attendance.toFixed(2)}` : ''
        ].filter(Boolean);
        return parts.join('<br>');
    }

    function renderClusterLegend() {
        const legend = document.getElementById('cluster-legend');
        legend.innerHTML = '<h3>Clusters Overview</h3>';

        // Sort by cluster number
        const sorted = [...STATE.clusters].sort((a, b) => a.cluster - b.cluster);

        sorted.forEach(c => {
            const div = document.createElement('div');
            div.className = 'cluster-legend-item';
            div.innerHTML = `
                <strong>Cluster ${c.cluster}</strong> —
                ${(c.pop_share * 100).toFixed(1)}% of ${STATE.metadata.analysis_universe.replace('_', ' ')} |
                Harris ${(c.vote_harris * 100).toFixed(0)}% / Trump ${(c.vote_trump * 100).toFixed(0)}% / Other ${(c.vote_other * 100).toFixed(0)}%
            `;
            legend.appendChild(div);
        });
    }

    // ========================================================================
    // QUIZ TAB
    // ========================================================================

    function setupQuizTab() {
        const container = document.getElementById('quiz-container');
        const submitBtn = document.getElementById('submit-quiz');

        // Build questions using only the 10 quiz features (selected by Random Forest)
        STATE.quizFeatures.forEach((varName, idx) => {
            const spec = STATE.preprocess.feature_specs[varName];
            const questionDiv = document.createElement('div');
            questionDiv.className = 'quiz-question';

            const title = document.createElement('h3');
            title.textContent = `${idx + 1}. ${getQuizQuestionText(varName)}`;
            questionDiv.appendChild(title);

            const optionsDiv = document.createElement('div');
            optionsDiv.className = 'quiz-options';

            // Generate scale
            const scale = getQuizScale(varName);
            scale.forEach(opt => {
                const label = document.createElement('label');
                label.className = 'quiz-option';

                const input = document.createElement('input');
                input.type = 'radio';
                input.name = varName;
                input.value = opt.value;

                label.appendChild(input);
                label.appendChild(document.createTextNode(opt.label));
                optionsDiv.appendChild(label);
            });

            questionDiv.appendChild(optionsDiv);
            container.appendChild(questionDiv);
        });

        submitBtn.classList.remove('hidden');

        submitBtn.addEventListener('click', () => {
            const answers = {};
            let complete = true;

            STATE.quizFeatures.forEach(varName => {
                const selected = document.querySelector(`input[name="${varName}"]:checked`);
                if (selected) {
                    answers[varName] = parseFloat(selected.value);
                } else {
                    complete = false;
                }
            });

            if (!complete) {
                alert('Please answer all questions before submitting.');
                return;
            }

            STATE.quizAnswers = answers;
            assignCluster();
            trackEvent('submitted_quiz');
        });
    }

    function assignCluster() {
        // Transform user answers to standardized space (using only 10 quiz features)
        const userVector = STATE.quizFeatures.map(varName => {
            let val = STATE.quizAnswers[varName];

            // Apply same transformations as training
            const spec = STATE.preprocess.feature_specs[varName];
            if (spec && spec.map) {
                val = spec.map[val] || val;
            }

            // Impute if missing (shouldn't happen if all answered)
            if (isNaN(val)) {
                val = STATE.preprocess.imputation_medians[varName];
            }

            // Standardize
            const mean = STATE.preprocess.scaling_means[varName];
            const std = STATE.preprocess.scaling_stds[varName];
            let standardized = (val - mean) / std;

            // Apply variance weighting if available (same as backend)
            if (STATE.preprocess.variance_weights && STATE.preprocess.variance_weights[varName]) {
                const weight = Math.sqrt(STATE.preprocess.variance_weights[varName]);
                standardized *= weight;
            }

            return standardized;
        });

        // Find nearest centroid (comparing only on the 10 quiz features)
        let minDist = Infinity;
        let assignedCluster = 0;

        Object.entries(STATE.centroids).forEach(([clusterId, centroid]) => {
            const centroidVector = STATE.quizFeatures.map(f => {
                let val = centroid[f];
                // Apply variance weighting to centroid too
                if (STATE.preprocess.variance_weights && STATE.preprocess.variance_weights[f]) {
                    const weight = Math.sqrt(STATE.preprocess.variance_weights[f]);
                    val *= weight;
                }
                return val;
            });
            const dist = euclideanDistance(userVector, centroidVector);
            if (dist < minDist) {
                minDist = dist;
                assignedCluster = parseInt(clusterId);
            }
        });

        displayQuizResults(assignedCluster);
    }

    function displayQuizResults(clusterId) {
        const resultsDiv = document.getElementById('quiz-results');
        resultsDiv.classList.remove('hidden');
        resultsDiv.scrollIntoView({ behavior: 'smooth' });

        const cluster = STATE.clusters.find(c => c.cluster === clusterId);
        const persona = STATE.avatars[clusterId];

        const infoDiv = document.getElementById('assigned-cluster-info');
        infoDiv.innerHTML = `
            <div class="cluster-card">
                <h4>You are closest to: ${persona.name} (Cluster ${clusterId})</h4>
                <p>${persona.description}</p>
                <div class="cluster-stats">
                    <div class="stat-item">
                        <strong>Population Share</strong>
                        <span>${(cluster.pop_share * 100).toFixed(1)}%</span>
                    </div>
                    <div class="stat-item">
                        <strong>Harris Support</strong>
                        <span>${(cluster.vote_harris * 100).toFixed(0)}%</span>
                    </div>
                    <div class="stat-item">
                        <strong>Trump Support</strong>
                        <span>${(cluster.vote_trump * 100).toFixed(0)}%</span>
                    </div>
                    <div class="stat-item">
                        <strong>Party ID Mean</strong>
                        <span>${cluster.partyid_mean ? cluster.partyid_mean.toFixed(2) : '?'}</span>
                    </div>
                </div>
                <p style="font-size:0.9rem;color:var(--color-text-muted);margin-top:10px;"><em>Party ID scale: 1 = Strong Democrat, 4 = Independent, 7 = Strong Republican. Range (p10–p90): ${cluster.partyid_p10 ? cluster.partyid_p10.toFixed(1) : '?'} – ${cluster.partyid_p90 ? cluster.partyid_p90.toFixed(1) : '?'}</em></p>
                <h5 style="margin-top:20px;">Demographics</h5>
                <p><strong>Gender:</strong> ${formatGenderSplit(cluster.demographics)}</p>
                <p><strong>Age:</strong> ${cluster.demographics.age_mean ? `Mean ${cluster.demographics.age_mean.toFixed(0)} years` : 'N/A'}</p>
                <p><strong>Education:</strong> ${cluster.demographics.education_college ? `${(cluster.demographics.education_college * 100).toFixed(0)}% college` : 'N/A'}</p>
                <p><strong>Religion:</strong> ${cluster.religion.mean_attendance ? `Attendance ${cluster.religion.mean_attendance.toFixed(2)} (1=never, 6=weekly+)` : 'N/A'}</p>
                <p><strong>Region:</strong> ${formatRegion(cluster.region)}</p>
            </div>
        `;

        // Find nearest 3 and farthest 1
        const distances = Object.entries(STATE.distances[clusterId])
            .filter(([id, dist]) => parseInt(id) !== clusterId)
            .sort((a, b) => a[1] - b[1]);

        const nearest = distances.slice(0, 3);
        const farthest = distances[distances.length - 1];

        const nearestDiv = document.getElementById('nearest-clusters');
        nearestDiv.innerHTML = nearest.map(([id, dist]) => {
            const c = STATE.clusters.find(cl => cl.cluster === parseInt(id));
            return `<div class="cluster-card">
                <h4>Cluster ${id} (distance: ${dist.toFixed(2)})</h4>
                <p><strong>Vote:</strong> Harris ${(c.vote_harris * 100).toFixed(0)}% / Trump ${(c.vote_trump * 100).toFixed(0)}% / Other ${(c.vote_other * 100).toFixed(0)}%</p>
                <p><strong>Party ID:</strong> ${c.partyid_mean ? c.partyid_mean.toFixed(1) : '?'} <em>(1=Strong Dem, 7=Strong Rep)</em></p>
                <p><strong>Gender:</strong> ${formatGenderSplit(c.demographics)}</p>
                <p><strong>Education:</strong> ${c.demographics.education_college ? `${(c.demographics.education_college * 100).toFixed(0)}% college` : 'N/A'}</p>
                <p><strong>Region:</strong> ${formatRegion(c.region)}</p>
            </div>`;
        }).join('');

        const farthestDiv = document.getElementById('farthest-cluster');
        const fC = STATE.clusters.find(c => c.cluster === parseInt(farthest[0]));
        farthestDiv.innerHTML = `<div class="cluster-card">
            <h4>Cluster ${farthest[0]} (distance: ${farthest[1].toFixed(2)})</h4>
            <p><strong>Vote:</strong> Harris ${(fC.vote_harris * 100).toFixed(0)}% / Trump ${(fC.vote_trump * 100).toFixed(0)}%</p>
            <p><strong>Party ID:</strong> ${fC.partyid_mean ? fC.partyid_mean.toFixed(1) : '?'} <em>(1=Strong Dem, 7=Strong Rep)</em></p>
            <p><strong>Gender:</strong> ${formatGenderSplit(fC.demographics)}</p>
            <p><strong>Education:</strong> ${fC.demographics.education_college ? `${(fC.demographics.education_college * 100).toFixed(0)}% college` : 'N/A'}</p>
            <p><strong>Region:</strong> ${formatRegion(fC.region)}</p>
        </div>`;
    }

    function formatGenderSplit(demo) {
        if (!demo.gender_Man) return 'N/A';
        return `${(demo.gender_Man * 100).toFixed(0)}% M / ${(demo.gender_Woman * 100).toFixed(0)}% W`;
    }

    function formatRegion(region) {
        if (!region.Northeast) return 'N/A';
        return `NE ${(region.Northeast * 100).toFixed(0)}% / MW ${(region.Midwest * 100).toFixed(0)}% / S ${(region.South * 100).toFixed(0)}% / W ${(region.West * 100).toFixed(0)}%`;
    }

    // ========================================================================
    // STORY MODE
    // ========================================================================

    function setupStoryTab() {
        const storyContainer = document.getElementById('story-container');
        const prevBtn = document.getElementById('story-prev');
        const nextBtn = document.getElementById('story-next');
        const indicator = document.getElementById('story-step-indicator');
        const titleEl = document.getElementById('story-title');

        titleEl.textContent = STATE.story.title;

        STATE.story.steps.forEach((step, idx) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = `story-step ${idx === 0 ? 'active' : ''}`;
            stepDiv.innerHTML = `
                <h3>${step.title}</h3>
                <p>${step.content}</p>
            `;
            storyContainer.appendChild(stepDiv);
        });

        function updateStory() {
            const steps = document.querySelectorAll('.story-step');
            steps.forEach((step, idx) => {
                step.classList.toggle('active', idx === STATE.currentStoryStep);
            });

            indicator.textContent = `Step ${STATE.currentStoryStep + 1} of ${STATE.story.steps.length}`;
            prevBtn.disabled = STATE.currentStoryStep === 0;
            nextBtn.disabled = STATE.currentStoryStep === STATE.story.steps.length - 1;
        }

        prevBtn.addEventListener('click', () => {
            if (STATE.currentStoryStep > 0) {
                STATE.currentStoryStep--;
                updateStory();
            }
        });

        nextBtn.addEventListener('click', () => {
            if (STATE.currentStoryStep < STATE.story.steps.length - 1) {
                STATE.currentStoryStep++;
                updateStory();
            }
        });

        updateStory();
    }

    // ========================================================================
    // PERSONA CHAT
    // ========================================================================

    function setupPersonaTab() {
        const select = document.getElementById('persona-select');
        const infoDiv = document.getElementById('persona-info');
        const chatMessages = document.getElementById('chat-messages');
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('chat-send');

        // Populate personas
        Object.entries(STATE.avatars).forEach(([clusterId, persona]) => {
            const opt = document.createElement('option');
            opt.value = clusterId;
            opt.textContent = `${persona.name} (Cluster ${clusterId})`;
            select.appendChild(opt);
        });

        function loadPersona() {
            const clusterId = select.value;
            STATE.currentPersona = parseInt(clusterId);
            const persona = STATE.avatars[clusterId];
            const cluster = STATE.clusters.find(c => c.cluster === parseInt(clusterId));

            // Build simplified profile display (basic info only)
            let profileHTML = `
                <h3>${persona.name}</h3>
                <p>${persona.description}</p>
                <p><strong>Cluster ${clusterId}</strong> — ${(cluster.pop_share * 100).toFixed(1)}% of voters</p>
                <p><strong>Vote:</strong> ${(cluster.vote_harris * 100).toFixed(0)}% Harris, ${(cluster.vote_trump * 100).toFixed(0)}% Trump, ${(cluster.vote_other * 100).toFixed(0)}% Other</p>
                <p><strong>Party ID:</strong> ${cluster.partyid_mean?.toFixed(1) || 'N/A'} (1=Strong Dem, 7=Strong Rep)</p>
            `;

            infoDiv.innerHTML = profileHTML;

            // Cluster distributions removed

            // Build collapsible policy positions section (below chat)
            const stancesDiv = document.getElementById('persona-stances');
            let stancesHTML = `
                <h4>All Policy Positions</h4>
                <ul class="stance-list">
            `;

            // Add all stances with scale range from the detail field
            for (const [varName, stance] of Object.entries(persona.stances)) {
                const decisiveStance = stance.decisive_stance || '';
                const detail = stance.detail || '';
                // Extract scale from detail, e.g. "Position: 2.5 on a 1-5 scale (1=Always, 5=Never)"
                const scaleMatch = detail.match(/on a (\d+)-(\d+) scale \(([^)]+)\)/);
                const value = stance.value?.toFixed(1) || 'N/A';
                const scaleRange = scaleMatch ? `${scaleMatch[1]}-${scaleMatch[2]}` : '';
                const scaleLabels = scaleMatch ? scaleMatch[3] : '';

                stancesHTML += `<li><strong>${decisiveStance}</strong> <span style="color:var(--color-text-muted);font-size:0.85em;">(${value} on ${scaleRange} scale: ${scaleLabels})</span></li>`;
            }

            stancesHTML += `</ul>`;
            stancesDiv.innerHTML = stancesHTML;

            // Clear chat
            STATE.chatHistory = [];
            chatMessages.innerHTML = '<p style="text-align:center;color:var(--color-text-muted);">Chat with this persona to understand their views. Try asking about current events or policy topics.</p>';

            trackEvent('opened_persona_chat', { cluster: clusterId });
        }

        // Toggle collapsible stances
        const toggleBtn = document.getElementById('toggle-stances');
        const stancesSection = document.getElementById('persona-stances');

        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                stancesSection.classList.toggle('hidden');
                if (stancesSection.classList.contains('hidden')) {
                    toggleBtn.textContent = '▼ View Full Policy Positions';
                } else {
                    toggleBtn.textContent = '▲ Hide Policy Positions';
                }
            });
        }

        select.addEventListener('change', loadPersona);
        loadPersona();

        async function sendMessage() {
            const msg = chatInput.value.trim();
            if (!msg) return;

            // Add user message
            addChatMessage('user', msg);
            chatInput.value = '';

            // Disable input while generating
            chatInput.disabled = true;
            sendBtn.disabled = true;

            try {
                // Try LLM first
                const llmResponse = await generateLLMResponse(msg);
                const response = llmResponse || generatePersonaResponse(msg);

                if (!response) {
                    // Both LLM and rule-based failed - show error
                    setTimeout(() => {
                        addChatMessage('assistant', '❌ Error: LLM backend is not responding. Please check that the server is running and the API key is configured. Try refreshing the page or restarting the server.');
                    }, 300);
                    chatInput.disabled = false;
                    sendBtn.disabled = false;
                    return;
                }

                setTimeout(() => {
                    addChatMessage('assistant', response.text, response.evidence, response.value);
                }, 300);

                trackEvent('persona_chat_message_sent', { used_llm: !!llmResponse });
            } finally {
                // Re-enable input
                chatInput.disabled = false;
                sendBtn.disabled = false;
                chatInput.focus();
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    }

    function addChatMessage(role, text, evidence = null, value = null) {
        const chatMessages = document.getElementById('chat-messages');
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${role}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = text;

        if (evidence) {
            const badge = document.createElement('span');
            badge.className = `evidence-badge ${evidence}`;
            badge.textContent = evidence.replace('_', ' ');
            badge.title = getEvidenceBadgeTooltip(evidence);
            bubble.appendChild(document.createTextNode(' '));
            bubble.appendChild(badge);

            if (evidence !== 'fictional_extrapolation' && value !== null) {
                const showBtn = document.createElement('button');
                showBtn.className = 'show-numbers-btn';
                showBtn.textContent = 'Show details ▼';

                const detailsDiv = document.createElement('div');
                detailsDiv.className = 'stance-details hidden';
                detailsDiv.style.marginTop = '10px';

                showBtn.addEventListener('click', () => {
                    // Find all related questions for this topic
                    const topicName = text.toLowerCase();
                    const persona = STATE.avatars[STATE.currentPersona];
                    let relatedStances = [];

                    for (const [varName, stance] of Object.entries(persona.stances)) {
                        const topicWords = varName.toLowerCase().split('_');
                        // Check if this stance is related to the user's query
                        if (topicWords.some(word => topicName.includes(word) || word.length > 4 && topicName.includes(word.substring(0, word.length - 1)))) {
                            // Extract the actual variable code (e.g., V241228 from ideology_V241228)
                            const actualVarCode = stance.variable || varName.match(/V\d+x?/)?.[0] || varName;

                            // Get readable name from question mappings
                            const readableName = getReadableFeatureName(actualVarCode);
                            const scaleInfo = getScaleInfo(actualVarCode);
                            relatedStances.push({
                                topic: readableName,
                                value: stance.value,
                                stance: stance.decisive_stance,
                                scale: scaleInfo
                            });
                        }
                    }

                    // Toggle details
                    if (detailsDiv.classList.contains('hidden')) {
                        // Build details HTML
                        let html = '<div style="background: var(--color-bg); padding: 10px; border-radius: 6px; margin-top: 8px;">';
                        html += '<strong>Related Policy Positions:</strong><br><br>';
                        relatedStances.forEach(s => {
                            html += `<div style="margin-bottom: 8px; padding: 8px; background: white; border-radius: 4px;">`;
                            html += `<strong>${s.topic}</strong><br>`;
                            html += `<small>Value: ${s.value.toFixed(2)} (${s.scale})</small><br>`;
                            html += `<small style="color: var(--color-text-muted);">${s.stance}</small>`;
                            html += `</div>`;
                        });
                        html += '</div>';
                        detailsDiv.innerHTML = html;
                        detailsDiv.classList.remove('hidden');
                        showBtn.textContent = 'Hide details ▲';
                    } else {
                        detailsDiv.classList.add('hidden');
                        showBtn.textContent = 'Show details ▼';
                    }

                    trackEvent('clicked_show_numbers', { evidence, related_count: relatedStances.length });
                });
                bubble.appendChild(document.createElement('br'));
                bubble.appendChild(showBtn);
                bubble.appendChild(detailsDiv);
            }
        }

        msgDiv.appendChild(bubble);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function generateLLMResponse(userMessage) {
        const llmEnabled = document.getElementById('enable-live-llm').checked;

        console.log('[LLM] Enabled:', llmEnabled);

        if (!llmEnabled) {
            console.log('[LLM] Disabled - returning null');
            return null; // Fall back to rule-based
        }

        const persona = STATE.avatars[STATE.currentPersona];
        const cluster = STATE.clusters.find(c => c.cluster === STATE.currentPersona);

        console.log('[LLM] Calling /api/chat...');

        try {
            // Call the Flask backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: [
                        ...STATE.chatHistory,
                        { role: 'user', content: userMessage }
                    ],
                    persona: persona,
                    provider: 'openai'  // or 'anthropic'
                })
            });

            if (!response.ok) {
                const error = await response.json();
                console.error('LLM Proxy error:', error);
                return null; // Fall back to rule-based
            }

            const data = await response.json();
            const llmText = data.response.trim();

            // Update chat history
            STATE.chatHistory.push({ role: 'user', content: userMessage });
            STATE.chatHistory.push({ role: 'assistant', content: llmText });

            // Keep history manageable (last 10 messages)
            if (STATE.chatHistory.length > 20) {
                STATE.chatHistory = STATE.chatHistory.slice(-20);
            }

            // Try to match LLM response to known stances for evidence badge
            const msg = userMessage.toLowerCase();
            let evidenceType = 'fictional_extrapolation';
            let value = null;

            for (const [topic, stance] of Object.entries(persona.stances)) {
                if (msg.includes(topic)) {
                    evidenceType = stance.evidence_type;
                    value = stance.value;
                    break;
                }
            }

            return {
                text: llmText,
                evidence: evidenceType,
                value: value
            };

        } catch (error) {
            console.error('[LLM] API call failed:', error);
            console.error('[LLM] Error message:', error.message);
            console.error('[LLM] Error stack:', error.stack);
            return null; // Fall back to rule-based
        }
    }

    function generatePersonaResponse(userMessage) {
        const persona = STATE.avatars[STATE.currentPersona];
        const msg = userMessage.toLowerCase();

        // Handle greetings
        if (msg.match(/^(hi|hello|hey|greetings|good morning|good afternoon|good evening)/i)) {
            return {
                text: `Hello! I'm ${persona.name}. Ask me about my views on political issues like abortion, immigration, healthcare, the environment, and more.`,
                evidence: null,
                value: null
            };
        }

        // Match keywords to topics
        const topicMatches = {
            'abortion': ['abortion', 'pro-life', 'pro-choice', 'roe', 'reproductive'],
            'immigration': ['immigration', 'immigrant', 'border', 'wall', 'migrant', 'undocumented'],
            'redistribution': ['government', 'services', 'spending', 'welfare', 'taxes'],
            'environment': ['environment', 'climate', 'carbon', 'green', 'pollution', 'warming'],
            'health': ['healthcare', 'health care', 'insurance', 'obamacare', 'medicare', 'medicaid'],
            'defense': ['defense', 'military', 'spending', 'army', 'navy', 'armed forces'],
            'welfare': ['welfare', 'assistance', 'benefits', 'food stamps', 'snap'],
            'efficacy_trust': ['trust', 'politicians', 'government', 'officials', 'corruption', 'washington'],
            'democracy': ['democracy', 'democratic', 'election', 'voting', 'vote'],
            'vaccines': ['vaccine', 'vaccination', 'immunization', 'vax'],
            'ideology': ['ideology', 'liberal', 'conservative', 'political views', 'left', 'right'],
            'religious_attendance': ['religion', 'religious', 'church', 'god', 'faith', 'worship', 'services'],
            'guaranteed_jobs': ['jobs', 'employment', 'unemployment', 'work', 'job guarantee'],
            'gay_marriage': ['gay', 'same-sex', 'marriage', 'lgbtq', 'lgbt'],
            'israel_aid': ['israel', 'israeli', 'middle east', 'palestine'],
            'ukraine_aid': ['ukraine', 'ukrainian', 'russia', 'putin', 'zelensky']
        };

        let matchedTopic = null;
        let bestMatchCount = 0;

        for (const [topic, keywords] of Object.entries(topicMatches)) {
            const matchCount = keywords.filter(kw => msg.includes(kw)).length;
            if (matchCount > bestMatchCount) {
                matchedTopic = topic;
                bestMatchCount = matchCount;
            }
        }

        if (matchedTopic && persona.stances[matchedTopic]) {
            const stance = persona.stances[matchedTopic];
            return {
                text: `${stance.decisive_stance} ${stance.detail}`,
                evidence: stance.evidence_type,
                value: stance.value
            };
        }

        // No fallback - LLM must handle fictional extrapolation
        // If we're here, the LLM failed or is disabled
        return null;
    }

    function updateClusterDistributions(currentCluster) {
        const displayDiv = document.getElementById('distribution-display');
        const checkboxes = document.querySelectorAll('.topic-checkbox');

        // Get selected topics
        const selectedTopics = Array.from(checkboxes)
            .filter(cb => cb.checked)
            .map(cb => ({ var: cb.value, label: cb.parentElement.textContent.trim() }));

        if (selectedTopics.length === 0) {
            displayDiv.innerHTML = '<p style="text-align:center;color:var(--color-text-muted);">Select topics above to see distributions</p>';
            return;
        }

        let html = '<div class="distribution-grid">';

        selectedTopics.forEach(topic => {
            const centroidKey = `centroid_${topic.var}`;
            const clusterValue = currentCluster[centroidKey];

            if (clusterValue === undefined || clusterValue === null) {
                return; // Skip if not available
            }

            // Denormalize from [-1, 1] back to [1, 7] scale
            // Assuming normalization was: (value - 4) / 3
            // So denormalization is: value * 3 + 4
            const denormalizedValue = (clusterValue * 3) + 4;

            // Clamp to [1, 7] range to handle any edge cases
            const originalScaleValue = Math.max(1, Math.min(7, denormalizedValue));

            // Position on 1-7 scale for visual display (0-100%)
            const position = ((originalScaleValue - 1) / 6) * 100;

            // Get question text for scale interpretation
            const questionText = getQuizQuestionText(topic.var);

            // Determine political lean based on position (1=liberal, 7=conservative for most questions)
            let interpretation = '';
            if (originalScaleValue < 3.33) {
                interpretation = 'More Liberal';
            } else if (originalScaleValue > 4.67) {
                interpretation = 'More Conservative';
            } else {
                interpretation = 'Moderate';
            }

            html += `
                <div class="distribution-item">
                    <h5>${topic.label}</h5>
                    <p class="distribution-question">${questionText}</p>
                    <div class="distribution-bar">
                        <div class="distribution-marker" style="left: ${position}%"></div>
                        <div class="distribution-track">
                            <span class="track-label left">1 (Liberal)</span>
                            <span class="track-label right">7 (Conservative)</span>
                        </div>
                    </div>
                    <p class="distribution-value">
                        <strong>This cluster:</strong> ${originalScaleValue.toFixed(2)} on 1-7 scale — ${interpretation}
                    </p>
                </div>
            `;
        });

        html += '</div>';
        displayDiv.innerHTML = html;
    }

    // Add event listeners for topic checkboxes (needs to run after DOM load)
    function setupDistributionListeners() {
        const checkboxes = document.querySelectorAll('.topic-checkbox');
        checkboxes.forEach(cb => {
            cb.addEventListener('change', () => {
                const cluster = STATE.clusters.find(c => c.cluster === STATE.currentPersona);
                if (cluster) {
                    updateClusterDistributions(cluster);
                }
            });
        });
    }

    // ========================================================================
    // WELCOME TOUR
    // ========================================================================

    function setupWelcomeTour() {
        const overlay = document.getElementById('welcome-overlay');
        const title = document.getElementById('tour-title');
        const content = document.getElementById('tour-content');
        const prevBtn = document.getElementById('tour-prev');
        const nextBtn = document.getElementById('tour-next');
        const closeBtn = document.getElementById('tour-close');
        const dontShowCheck = document.getElementById('tour-dont-show');
        const indicator = document.getElementById('tour-step-indicator');
        const replayBtn = document.getElementById('replay-tour-btn');

        let currentStep = 0;
        const tour = STATE.welcomeTour.tours[0];
        const steps = tour.steps;

        function showTour() {
            if (localStorage.getItem('tourCompleted') === 'true') return;
            overlay.classList.remove('hidden');
            updateTourStep();
            trackEvent('tour_started');
        }

        function updateTourStep() {
            const step = steps[currentStep];
            title.textContent = `${tour.title} (${currentStep + 1}/${steps.length})`;
            content.textContent = step.content;
            indicator.textContent = `${currentStep + 1} / ${steps.length}`;

            prevBtn.disabled = currentStep === 0;
            nextBtn.textContent = currentStep === steps.length - 1 ? 'Finish' : 'Next';

            // Navigate to relevant tab if specified
            if (step.target) {
                if (step.target === '#explore-tab') window.switchToTab('explore');
                else if (step.target === '#quiz-tab') window.switchToTab('quiz');
                else if (step.target === '#story-tab') window.switchToTab('story');
                else if (step.target === '#persona-tab') window.switchToTab('persona');
            }
        }

        function closeTour() {
            overlay.classList.add('hidden');
            if (dontShowCheck.checked) {
                localStorage.setItem('tourCompleted', 'true');
            }
            // Go back to explore tab
            window.switchToTab('explore');
            trackEvent('tour_completed');
        }

        prevBtn.addEventListener('click', () => {
            if (currentStep > 0) {
                currentStep--;
                updateTourStep();
            }
        });

        nextBtn.addEventListener('click', () => {
            if (currentStep < steps.length - 1) {
                currentStep++;
                updateTourStep();
            } else {
                closeTour();
            }
        });

        closeBtn.addEventListener('click', closeTour);

        replayBtn.addEventListener('click', () => {
            currentStep = 0;
            overlay.classList.remove('hidden');
            updateTourStep();
            trackEvent('tour_replayed');
        });

        // Show tour on first visit
        setTimeout(showTour, 1000);
    }

    // ========================================================================
    // ANALYTICS (Privacy-preserving event tracking)
    // ========================================================================

    function trackEvent(eventName, props = {}) {
        console.log('Analytics event:', eventName, props);

        // Example: Plausible Analytics (privacy-first, no cookies)
        // Uncomment and configure after setup:
        /*
        if (window.plausible) {
            window.plausible(eventName, { props });
        }
        */

        // Example: Custom endpoint (Cloudflare Worker, etc.)
        /*
        fetch('/api/analytics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ event: eventName, props, timestamp: new Date().toISOString() })
        }).catch(err => console.error('Analytics error:', err));
        */
    }

    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================

    function getReadableFeatureName(varName) {
        // Use concise axis labels for dimension dropdowns
        if (STATE.questionMappings && STATE.questionMappings.axis_labels) {
            return STATE.questionMappings.axis_labels[varName] || varName;
        }
        // Fallback to variable name if mappings not loaded yet
        return varName;
    }

    function getQuizQuestionText(varName) {
        // Use mappings from question_mappings.json if available
        if (STATE.questionMappings && STATE.questionMappings.quiz_questions) {
            return STATE.questionMappings.quiz_questions[varName] || varName;
        }
        // Fallback to variable name if mappings not loaded yet
        return varName;
    }

    function getScaleInfo(varName) {
        // Extract scale information from question text
        const questionText = getQuizQuestionText(varName);

        // Look for patterns like (1 = X, N = Y) where N is any number
        const scaleMatch = questionText.match(/\(1\s*=\s*([^,)]+),\s*(\d+)\s*=\s*([^)]+)\)/);
        if (scaleMatch) {
            const lowEndpoint = scaleMatch[1].trim();
            const highValue = parseInt(scaleMatch[2]);
            const highEndpoint = scaleMatch[3].trim();

            // Always show endpoints for clarity
            return `${lowEndpoint} (1) to ${highEndpoint} (${highValue})`;
        }

        // Fallback if pattern doesn't match
        return '1-7 scale';
    }

    function getQuizScale(varName) {
        // Return appropriate scale for each question
        if (varName === 'V241732') {
            return [
                { value: 1, label: 'Favor' },
                { value: 3, label: 'Neither' },
                { value: 2, label: 'Oppose' }
            ];
        } else if (varName === 'V241726') {
            return [
                { value: 1, label: 'Strongly agree' },
                { value: 2, label: 'Agree' },
                { value: 3, label: 'Neither' },
                { value: 4, label: 'Disagree' },
                { value: 5, label: 'Strongly disagree' }
            ];
        } else if (varName === 'V241731') {
            return [
                { value: 1, label: 'Extremely important' },
                { value: 2, label: 'Very important' },
                { value: 3, label: 'Moderately important' },
                { value: 4, label: 'Not at all important' }
            ];
        } else if (varName === 'V241386') {
            return [
                { value: 1, label: 'Felony/deport' },
                { value: 2, label: '2' },
                { value: 3, label: '3' },
                { value: 4, label: '4' },
                { value: 5, label: 'No penalty' }
            ];
        } else {
            // Default 7-point scale
            return Array.from({ length: 7 }, (_, i) => ({
                value: i + 1,
                label: `${i + 1}`
            }));
        }
    }

    function euclideanDistance(a, b) {
        return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
    }

    function getEvidenceBadgeTooltip(evidence) {
        const tooltips = {
            'observed': 'This stance is directly observed in ANES survey data for this cluster.',
            'inferred_by_ml': 'This stance is predicted by machine learning from related survey questions.',
            'fictional_extrapolation': 'This topic is not measured in ANES; the response is a fictional extrapolation.'
        };
        return tooltips[evidence] || '';
    }

    // ========================================================================
    // ENTRY POINT
    // ========================================================================

    document.addEventListener('DOMContentLoaded', () => {
        loadAllData();
    });

})();
