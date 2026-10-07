// CI/CD pipeline. Requires a Jenkins agent with Python 3.12, Docker and
// docker compose available.

pipeline {
  agent any

  options {
    timestamps()
    timeout(time: 30, unit: 'MINUTES')
    disableConcurrentBuilds()
  }

  environment {
    IMAGE_API    = "fraud-detection/api"
    IMAGE_STREAM = "fraud-detection/stream"
    TAG          = "${env.GIT_COMMIT ? env.GIT_COMMIT.take(8) : env.BUILD_NUMBER}"
    REGISTRY     = "ghcr.io/hardikmehta1122"
  }

  stages {

    stage('Install') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt -r requirements-stream.txt
          pip install ruff pytest
        '''
      }
    }

    stage('Lint') {
      steps {
        sh '''
          . .venv/bin/activate
          ruff check pipeline model backend streaming data
        '''
      }
    }

    stage('Unit tests') {
      steps {
        sh '''
          . .venv/bin/activate
          pytest -q tests
        '''
      }
    }

    stage('Build pipeline artifacts') {
      steps {
        sh '''
          . .venv/bin/activate
          python data/generate_data.py
          python -m pipeline.ingest
          python -m model.train
          python -m backend.build_serving_db
        '''
      }
    }

    stage('Leakage + experiment checks') {
      steps {
        sh '''
          . .venv/bin/activate
          python -m model.experiments
          python -m model.leakage_demo
          python -m model.validation_timing_leakage_demo
        '''
      }
    }

    stage('Build images') {
      steps {
        sh '''
          docker build -f docker/Dockerfile.api    -t ${IMAGE_API}:${TAG}    .
          docker build -f docker/Dockerfile.stream -t ${IMAGE_STREAM}:${TAG} .
        '''
      }
    }

    stage('Compose smoke test') {
      steps {
        sh '''
          docker compose up -d redpanda
          docker compose up -d
          sleep 45
          curl -sf http://localhost:8000/api/stats
          curl -sf http://localhost:8000/metrics > /dev/null
          curl -sf http://localhost:9100/metrics > /dev/null   # faust worker
          curl -sf http://localhost:9090/-/healthy > /dev/null # prometheus
        '''
      }
      post {
        always {
          sh 'docker compose logs --no-color > compose-logs.txt || true'
          archiveArtifacts artifacts: 'compose-logs.txt', allowEmptyArchive: true
          sh 'docker compose down -v || true'
        }
      }
    }

    stage('Push images') {
      when { branch 'main' }
      steps {
        withCredentials([usernamePassword(
            credentialsId: 'ghcr-credentials',
            usernameVariable: 'REG_USER',
            passwordVariable: 'REG_PASS')]) {
          sh '''
            echo "$REG_PASS" | docker login ghcr.io -u "$REG_USER" --password-stdin
            docker tag ${IMAGE_API}:${TAG}    ${REGISTRY}/${IMAGE_API}:${TAG}
            docker tag ${IMAGE_STREAM}:${TAG} ${REGISTRY}/${IMAGE_STREAM}:${TAG}
            docker push ${REGISTRY}/${IMAGE_API}:${TAG}
            docker push ${REGISTRY}/${IMAGE_STREAM}:${TAG}
          '''
        }
      }
    }
  }

  post {
    always  { cleanWs() }
    success { echo "Build ${env.BUILD_NUMBER} OK (${env.TAG})" }
    failure { echo "Build ${env.BUILD_NUMBER} FAILED" }
  }
}
